#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 control node: điều khiển robot bằng mục tiêu số xung encoder.

Khác với bản cũ:
- Không dùng PID Z để tạo PWM rồi mới quan sát encoder.
- Không dùng chênh lệch encoder trái - phải để bù bánh.
- Mỗi chu kỳ lập kế hoạch sẽ:
    1) Đo depth và error_x.
    2) Tính quãng đường tịnh tiến cần đi.
    3) Đổi error_x sang góc yaw cần quay.
    4) Tính quãng đường riêng của bánh trái và bánh phải.
    5) Đổi từng quãng đường thành số xung mục tiêu riêng.
    6) Chạy từng bánh cho đến khi số xung phát sinh trong đoạn làm việc
       đạt mục tiêu của chính bánh đó.

Dữ liệu encoder ESP32:
    ENC,left,right

Dữ liệu tracking YOLO:
    [locked, error_x, error_y, depth_mm, lost_count, mode_id]
"""

import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray

try:
    import serial
except ImportError as exc:
    raise RuntimeError(
        'Chưa cài pyserial. Chạy: sudo apt install python3-serial'
    ) from exc


# ============================================================
# SETTING CONFIG - CHỈNH CÁC GIÁ TRỊ Ở KHU VỰC NÀY
# ============================================================

# ---------------- ROS / UART ----------------
TARGET_TOPIC = '/target_tracking'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
SERIAL_PORT = '/dev/ttyUSB0'
BAUDRATE = 115200
CONTROL_HZ = 50.0
LOG_INTERVAL_S = 0.25
UART_RECONNECT_INTERVAL_S = 2.0
UART_STARTUP_DELAY_S = 0.10
UART_STOP_DELAY_S = 0.05
CMD_HEARTBEAT_S = 0.06

# ---------------- Vị trí dữ liệu /target_tracking ----------------
# YOLO publish: [locked, error_x, error_y, depth_mm, lost_count, mode_id]
LOCKED_INDEX = 0
ERROR_X_INDEX = 1
DEPTH_INDEX = 3
LOST_INDEX = 4
MODE_INDEX = 5

# ---------------- An toàn tracking / depth ----------------
TRACKING_TIMEOUT_S = 0.50
MAX_LOST_FRAMES = 2
MIN_VALID_DEPTH_MM = 300.0
MAX_VALID_DEPTH_MM = 4000.0
ENCODER_TIMEOUT_S = 0.25

# ---------------- Vùng giữ khoảng cách ----------------
# depth < 1500 mm: lập kế hoạch lùi tới biên 1500 mm
# 1500..1800 mm: không tịnh tiến
# depth > 1800 mm: lập kế hoạch tiến tới biên 1800 mm
STOP_DISTANCE_NEAR_MM = 1500.0
STOP_DISTANCE_FAR_MM = 1800.0

# Chỉ lập một đoạn ngắn rồi đo lại camera và lập kế hoạch tiếp.
# Đoạn ngắn giúp robot không chạy theo một kế hoạch cũ khi người đang di chuyển.
MAX_LINEAR_SEGMENT_MM = 250.0
MIN_LINEAR_PLAN_MM = 20.0

# ---------------- Camera / quy đổi error_x sang yaw ----------------
# Ưu tiên fx nhận từ /camera/color/camera_info.
# Nếu chưa nhận CameraInfo thì dùng giá trị dự phòng này.
CAMERA_FX_FALLBACK_PX = 570.0
X_DEADZONE_PX = 45.0

# Mỗi kế hoạch chỉ quay tối đa từng này độ rồi đo lại.
MAX_YAW_SEGMENT_DEG = 7.0
MIN_YAW_PLAN_DEG = 0.6

# Khi target lệch quá xa thì chỉ quay, chưa tiến/lùi.
YAW_ONLY_THRESHOLD_PX = 190.0

# error_x > 0 là target bên phải.
# Với mô hình động học chuẩn, target bên phải cần yaw âm.
# Đổi giữa -1.0 và +1.0 nếu phần cứng quay ngược.
YAW_SIGN = -1.0

# Khi vừa đi vừa cua, không để cung quay lớn hơn tỷ lệ này so với
# quãng đường tịnh tiến. Mục đích: không làm một bánh đổi chiều đột ngột.
MAX_TURN_ARC_TO_LINEAR_RATIO = 0.45

# ---------------- Thông số cơ khí ----------------
ENCODER_PULSES_PER_REV = 225.0
WHEEL_DIAMETER_MM = 65.0
WHEEL_TRACK_MM = 225.0       # khoảng cách tâm hai bánh

# ESP32 hiện gửi encoder trong miền 0..32767; cấu hình cũ có thể về 0 ở 32000.
ENCODER_MAX_COUNT = 32767
ENCODER_COUNT_MODULO = 32768
ENCODER_ALT_COUNT_MODULO = 32001
ENCODER_MAX_VALID_DELTA = 1500

# ---------------- Điều khiển theo mục tiêu xung ----------------
# Lệnh chạy xa mục tiêu và lệnh chạy gần mục tiêu.
PULSE_CRUISE_CMD = 105
PULSE_APPROACH_CMD = 18

# Còn dưới số xung này thì giảm dần lệnh motor.
PULSE_SLOWDOWN_REMAINING = 90.0

# Xem như bánh đã đủ xung khi số xung còn lại <= biên này.
PULSE_STOP_MARGIN = 3.0

# Không tạo kế hoạch nhỏ hơn số xung này.
PULSE_MIN_TARGET = 3.0

# Dự phòng vọt lố: mục tiêu thực thi được giảm trước từng này xung.
# Tăng nếu xe vẫn chạy quá biên; giảm nếu xe dừng thiếu.
PULSE_BRAKE_MARGIN = 4.0

# Một đoạn chạy quá lâu mà chưa đủ xung thì dừng an toàn và lập lại.
PLAN_TIMEOUT_S = 4.0
PLAN_SETTLE_S = 0.08
PLAN_MIN_RUN_BEFORE_REPLAN_S = 0.12

# Nếu yaw đã đổi dấu hoặc đã về giữa ảnh thì hủy đoạn cũ để tránh quay quá góc.
REPLAN_ON_YAW_SIGN_CHANGE = True
REPLAN_WHEN_X_CENTERED = True

# ---------------- Giới hạn lệnh động cơ ----------------
MAX_OUTPUT_PWM = 190
MAX_PWM_STEP = 10

# Theo yêu cầu trước: lệnh hai bánh không chênh nhau quá 50.
# Khi quay tại chỗ, ví dụ +25 và -25, độ chênh là 50.
MAX_WHEEL_CMD_DIFFERENCE = 80


class EncoderPulseTargetControlNode(Node):
    """Điều khiển theo mục tiêu số xung riêng của từng bánh."""

    ENCODER_MODULO = ENCODER_COUNT_MODULO
    ENCODER_ALT_MODULO = ENCODER_ALT_COUNT_MODULO

    def __init__(self) -> None:
        super().__init__('control_z_node')

        # ============================================================
        # Khai báo ROS parameters
        # ============================================================
        self.declare_parameter('target_topic', TARGET_TOPIC)
        self.declare_parameter('camera_info_topic', CAMERA_INFO_TOPIC)
        self.declare_parameter('serial_port', SERIAL_PORT)
        self.declare_parameter('baudrate', BAUDRATE)
        self.declare_parameter('control_hz', CONTROL_HZ)

        self.declare_parameter('tracking_timeout_s', TRACKING_TIMEOUT_S)
        self.declare_parameter('max_lost_frames', MAX_LOST_FRAMES)
        self.declare_parameter('min_valid_depth_mm', MIN_VALID_DEPTH_MM)
        self.declare_parameter('max_valid_depth_mm', MAX_VALID_DEPTH_MM)
        self.declare_parameter('encoder_timeout_s', ENCODER_TIMEOUT_S)

        self.declare_parameter('stop_distance_near_mm', STOP_DISTANCE_NEAR_MM)
        self.declare_parameter('stop_distance_far_mm', STOP_DISTANCE_FAR_MM)
        self.declare_parameter('max_linear_segment_mm', MAX_LINEAR_SEGMENT_MM)
        self.declare_parameter('min_linear_plan_mm', MIN_LINEAR_PLAN_MM)

        self.declare_parameter('camera_fx_fallback_px', CAMERA_FX_FALLBACK_PX)
        self.declare_parameter('x_deadzone_px', X_DEADZONE_PX)
        self.declare_parameter('max_yaw_segment_deg', MAX_YAW_SEGMENT_DEG)
        self.declare_parameter('min_yaw_plan_deg', MIN_YAW_PLAN_DEG)
        self.declare_parameter('yaw_only_threshold_px', YAW_ONLY_THRESHOLD_PX)
        self.declare_parameter('yaw_sign', YAW_SIGN)
        self.declare_parameter(
            'max_turn_arc_to_linear_ratio',
            MAX_TURN_ARC_TO_LINEAR_RATIO,
        )

        self.declare_parameter('encoder_pulses_per_rev', ENCODER_PULSES_PER_REV)
        self.declare_parameter('wheel_diameter_mm', WHEEL_DIAMETER_MM)
        self.declare_parameter('wheel_track_mm', WHEEL_TRACK_MM)

        self.declare_parameter('pulse_cruise_cmd', PULSE_CRUISE_CMD)
        self.declare_parameter('pulse_approach_cmd', PULSE_APPROACH_CMD)
        self.declare_parameter(
            'pulse_slowdown_remaining',
            PULSE_SLOWDOWN_REMAINING,
        )
        self.declare_parameter('pulse_stop_margin', PULSE_STOP_MARGIN)
        self.declare_parameter('pulse_min_target', PULSE_MIN_TARGET)
        self.declare_parameter('pulse_brake_margin', PULSE_BRAKE_MARGIN)
        self.declare_parameter('plan_timeout_s', PLAN_TIMEOUT_S)
        self.declare_parameter('plan_settle_s', PLAN_SETTLE_S)
        self.declare_parameter(
            'plan_min_run_before_replan_s',
            PLAN_MIN_RUN_BEFORE_REPLAN_S,
        )
        self.declare_parameter(
            'replan_on_yaw_sign_change',
            REPLAN_ON_YAW_SIGN_CHANGE,
        )
        self.declare_parameter(
            'replan_when_x_centered',
            REPLAN_WHEN_X_CENTERED,
        )

        self.declare_parameter('max_output_pwm', MAX_OUTPUT_PWM)
        self.declare_parameter('max_pwm_step', MAX_PWM_STEP)
        self.declare_parameter(
            'max_wheel_cmd_difference',
            MAX_WHEEL_CMD_DIFFERENCE,
        )
        self.declare_parameter('cmd_heartbeat_s', CMD_HEARTBEAT_S)

        self.declare_parameter('locked_index', LOCKED_INDEX)
        self.declare_parameter('error_x_index', ERROR_X_INDEX)
        self.declare_parameter('depth_index', DEPTH_INDEX)
        self.declare_parameter('lost_index', LOST_INDEX)
        self.declare_parameter('mode_index', MODE_INDEX)

        # ============================================================
        # Đọc parameters
        # ============================================================
        self.target_topic = str(self.get_parameter('target_topic').value)
        self.camera_info_topic = str(
            self.get_parameter('camera_info_topic').value
        )
        self.serial_port = str(self.get_parameter('serial_port').value)
        self.baudrate = int(self.get_parameter('baudrate').value)
        self.control_hz = max(10.0, float(
            self.get_parameter('control_hz').value
        ))

        self.tracking_timeout_s = max(0.05, float(
            self.get_parameter('tracking_timeout_s').value
        ))
        self.max_lost_frames = max(0, int(
            self.get_parameter('max_lost_frames').value
        ))
        self.min_valid_depth_mm = float(
            self.get_parameter('min_valid_depth_mm').value
        )
        self.max_valid_depth_mm = float(
            self.get_parameter('max_valid_depth_mm').value
        )
        self.encoder_timeout_s = max(0.05, float(
            self.get_parameter('encoder_timeout_s').value
        ))

        self.stop_distance_near_mm = float(
            self.get_parameter('stop_distance_near_mm').value
        )
        self.stop_distance_far_mm = float(
            self.get_parameter('stop_distance_far_mm').value
        )
        if self.stop_distance_far_mm <= self.stop_distance_near_mm:
            raise ValueError(
                'stop_distance_far_mm phải lớn hơn stop_distance_near_mm'
            )
        self.max_linear_segment_mm = max(1.0, float(
            self.get_parameter('max_linear_segment_mm').value
        ))
        self.min_linear_plan_mm = max(0.0, float(
            self.get_parameter('min_linear_plan_mm').value
        ))

        self.camera_fx_fallback_px = max(1.0, float(
            self.get_parameter('camera_fx_fallback_px').value
        ))
        self.camera_fx_px = self.camera_fx_fallback_px
        self.x_deadzone_px = max(0.0, float(
            self.get_parameter('x_deadzone_px').value
        ))
        self.max_yaw_segment_rad = math.radians(max(0.1, float(
            self.get_parameter('max_yaw_segment_deg').value
        )))
        self.min_yaw_plan_rad = math.radians(max(0.0, float(
            self.get_parameter('min_yaw_plan_deg').value
        )))
        self.yaw_only_threshold_px = max(self.x_deadzone_px, float(
            self.get_parameter('yaw_only_threshold_px').value
        ))
        self.yaw_sign = 1.0 if float(
            self.get_parameter('yaw_sign').value
        ) >= 0.0 else -1.0
        self.max_turn_arc_to_linear_ratio = self.clamp(float(
            self.get_parameter('max_turn_arc_to_linear_ratio').value
        ), 0.0, 1.0)

        self.encoder_pulses_per_rev = max(1.0, float(
            self.get_parameter('encoder_pulses_per_rev').value
        ))
        self.wheel_diameter_mm = max(1.0, float(
            self.get_parameter('wheel_diameter_mm').value
        ))
        self.wheel_track_mm = max(1.0, float(
            self.get_parameter('wheel_track_mm').value
        ))
        self.mm_per_pulse = (
            math.pi * self.wheel_diameter_mm
            / self.encoder_pulses_per_rev
        )

        self.pulse_cruise_cmd = int(self.clamp(float(
            self.get_parameter('pulse_cruise_cmd').value
        ), 1.0, 255.0))
        self.pulse_approach_cmd = int(self.clamp(float(
            self.get_parameter('pulse_approach_cmd').value
        ), 1.0, float(self.pulse_cruise_cmd)))
        self.pulse_slowdown_remaining = max(1.0, float(
            self.get_parameter('pulse_slowdown_remaining').value
        ))
        self.pulse_stop_margin = max(0.0, float(
            self.get_parameter('pulse_stop_margin').value
        ))
        self.pulse_min_target = max(1.0, float(
            self.get_parameter('pulse_min_target').value
        ))
        self.pulse_brake_margin = max(0.0, float(
            self.get_parameter('pulse_brake_margin').value
        ))
        self.plan_timeout_s = max(0.2, float(
            self.get_parameter('plan_timeout_s').value
        ))
        self.plan_settle_s = max(0.0, float(
            self.get_parameter('plan_settle_s').value
        ))
        self.plan_min_run_before_replan_s = max(0.0, float(
            self.get_parameter('plan_min_run_before_replan_s').value
        ))
        self.replan_on_yaw_sign_change = bool(
            self.get_parameter('replan_on_yaw_sign_change').value
        )
        self.replan_when_x_centered = bool(
            self.get_parameter('replan_when_x_centered').value
        )

        self.max_output_pwm = int(self.clamp(float(
            self.get_parameter('max_output_pwm').value
        ), 1.0, 255.0))
        self.max_pwm_step = max(1, int(
            self.get_parameter('max_pwm_step').value
        ))
        self.max_wheel_cmd_difference = max(0, int(
            self.get_parameter('max_wheel_cmd_difference').value
        ))
        self.cmd_heartbeat_s = self.clamp(float(
            self.get_parameter('cmd_heartbeat_s').value
        ), 0.02, 0.12)

        self.locked_index = int(self.get_parameter('locked_index').value)
        self.error_x_index = int(self.get_parameter('error_x_index').value)
        self.depth_index = int(self.get_parameter('depth_index').value)
        self.lost_index = int(self.get_parameter('lost_index').value)
        self.mode_index = int(self.get_parameter('mode_index').value)

        # ============================================================
        # Tracking state
        # ============================================================
        self.locked = False
        self.error_x = 0.0
        self.depth_mm = 0.0
        self.lost_frames = 0
        self.mode_id = 0
        self.last_tracking_time = 0.0

        # ============================================================
        # Encoder state
        # ============================================================
        self.encoder_left: Optional[int] = None
        self.encoder_right: Optional[int] = None
        self.previous_encoder_left: Optional[int] = None
        self.previous_encoder_right: Optional[int] = None
        self.delta_left = 0
        self.delta_right = 0
        self.encoder_last_time = 0.0
        self.rx_buffer = ''

        # Tổng xung chênh lệch theo từng lần đọc, chỉ để log.
        self.encoder_accumulated_left = 0.0
        self.encoder_accumulated_right = 0.0

        # ============================================================
        # Pulse plan state
        # ============================================================
        self.plan_active = False
        self.plan_start_time = 0.0
        self.plan_settle_until = 0.0
        self.plan_state = 'IDLE'

        self.plan_linear_mm = 0.0
        self.plan_yaw_rad = 0.0
        self.plan_depth_start_mm = 0.0
        self.plan_error_x_start = 0.0
        self.plan_yaw_sign = 0

        self.plan_left_distance_mm = 0.0
        self.plan_right_distance_mm = 0.0
        self.plan_left_direction = 0
        self.plan_right_direction = 0
        self.plan_left_target_pulses = 0.0
        self.plan_right_target_pulses = 0.0
        self.plan_left_progress_pulses = 0.0
        self.plan_right_progress_pulses = 0.0

        # ============================================================
        # Motor / UART state
        # ============================================================
        self.current_left_pwm = 0
        self.current_right_pwm = 0
        self.last_sent_left: Optional[int] = None
        self.last_sent_right: Optional[int] = None
        self.last_cmd_send_time = 0.0

        self.serial_conn: Optional[serial.Serial] = None
        self.last_reconnect_attempt = 0.0
        self.last_log_time = 0.0

        self.open_serial()

        self.subscription = self.create_subscription(
            Float32MultiArray,
            self.target_topic,
            self.tracking_callback,
            10,
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )
        self.timer = self.create_timer(
            1.0 / self.control_hz,
            self.control_loop,
        )

        self.get_logger().info(
            'ENCODER PULSE TARGET CONTROL | '
            f'stop={self.stop_distance_near_mm:.0f}..'
            f'{self.stop_distance_far_mm:.0f}mm | '
            f'{self.encoder_pulses_per_rev:.0f} pulse/rev | '
            f'{self.mm_per_pulse:.3f}mm/pulse | '
            f'track={self.wheel_track_mm:.1f}mm | '
            f'UART={self.serial_port}@{self.baudrate}'
        )

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def sign(value: float) -> int:
        if value > 0.0:
            return 1
        if value < 0.0:
            return -1
        return 0

    @staticmethod
    def approach(current: int, target: int, step: int) -> int:
        if target > current:
            return min(current + step, target)
        if target < current:
            return max(current - step, target)
        return current

    def encoder_delta_with_wrap(self, new_value: int, old_value: int) -> int:
        """Tính số xung phát sinh giữa hai lần đọc của cùng một encoder."""
        if new_value >= old_value:
            delta = new_value - old_value
            return delta if delta <= ENCODER_MAX_VALID_DELTA else 0

        candidates = [
            (new_value - old_value) % self.ENCODER_MODULO,
            (new_value - old_value) % self.ENCODER_ALT_MODULO,
        ]
        valid = [
            value for value in candidates
            if 0 <= value <= ENCODER_MAX_VALID_DELTA
        ]
        return min(valid) if valid else 0

    def encoder_valid(self, now: float) -> bool:
        return (
            self.encoder_left is not None
            and self.encoder_right is not None
            and self.encoder_last_time > 0.0
            and now - self.encoder_last_time <= self.encoder_timeout_s
        )

    def tracking_valid(self, now: float) -> tuple[bool, str]:
        if self.last_tracking_time <= 0.0:
            return False, 'WAIT_TRACKING'
        if now - self.last_tracking_time > self.tracking_timeout_s:
            return False, 'STOP_TRACK_TIMEOUT'
        if not self.locked:
            return False, 'STOP_NOT_LOCKED'
        if self.lost_frames > self.max_lost_frames:
            return False, 'STOP_LOST_TARGET'
        if not math.isfinite(self.depth_mm) or not math.isfinite(self.error_x):
            return False, 'STOP_DATA_NAN'
        if not (
            self.min_valid_depth_mm
            <= self.depth_mm
            <= self.max_valid_depth_mm
        ):
            return False, 'STOP_DEPTH_INVALID'
        return True, 'TRACKING_OK'

    def limit_wheel_cmd_difference(
        self,
        left: int,
        right: int,
    ) -> tuple[int, int]:
        """Giữ abs(left-right) không vượt cấu hình.

        Bánh đã đủ xung và có lệnh 0 phải tiếp tục giữ 0; không được nâng
        bánh đó chạy lại chỉ để giữ giá trị trung bình.
        """
        max_diff = self.max_wheel_cmd_difference
        left = int(left)
        right = int(right)

        if max_diff <= 0:
            return 0, 0

        difference = left - right
        if abs(difference) <= max_diff:
            return left, right

        # Một bánh đã hoàn thành mục tiêu: giữ bánh đó bằng 0 và chỉ giới hạn
        # bánh còn lại xuống max_diff.
        if left == 0:
            return 0, int(self.clamp(right, -max_diff, max_diff))
        if right == 0:
            return int(self.clamp(left, -max_diff, max_diff)), 0

        # Hai bánh ngược chiều (quay tại chỗ): co tỷ lệ cả hai để tổng độ lớn
        # không vượt max_diff.
        if left * right < 0:
            total = abs(left) + abs(right)
            if total <= 0:
                return 0, 0
            scale = max_diff / float(total)
            return int(round(left * scale)), int(round(right * scale))

        # Hai bánh cùng chiều: giữ gần nguyên giá trị trung bình.
        average = (left + right) / 2.0
        if difference > 0:
            limited_left = int(round(average + max_diff / 2.0))
            limited_right = limited_left - max_diff
        else:
            limited_right = int(round(average + max_diff / 2.0))
            limited_left = limited_right - max_diff
        return limited_left, limited_right

    # ============================================================
    # ROS callbacks
    # ============================================================
    def tracking_callback(self, msg: Float32MultiArray) -> None:
        data = msg.data
        required = max(
            self.locked_index,
            self.error_x_index,
            self.depth_index,
        )
        if len(data) <= required:
            self.get_logger().warning(
                f'/target_tracking chỉ có {len(data)} phần tử; '
                f'cần ít nhất {required + 1}'
            )
            return

        self.locked = bool(int(round(float(data[self.locked_index]))))
        self.error_x = float(data[self.error_x_index])
        self.depth_mm = float(data[self.depth_index])

        if 0 <= self.lost_index < len(data):
            self.lost_frames = int(round(float(data[self.lost_index])))
        else:
            self.lost_frames = 0

        if 0 <= self.mode_index < len(data):
            self.mode_id = int(round(float(data[self.mode_index])))
        else:
            self.mode_id = 0

        self.last_tracking_time = time.monotonic()

    def camera_info_callback(self, msg: CameraInfo) -> None:
        if len(msg.k) >= 1 and math.isfinite(msg.k[0]) and msg.k[0] > 1.0:
            self.camera_fx_px = float(msg.k[0])

    # ============================================================
    # UART
    # ============================================================
    def open_serial(self) -> None:
        self.last_reconnect_attempt = time.monotonic()
        try:
            self.serial_conn = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=0,
                write_timeout=0.1,
            )
            self.serial_conn.reset_input_buffer()
            self.serial_conn.reset_output_buffer()
            self.previous_encoder_left = None
            self.previous_encoder_right = None
            self.encoder_last_time = 0.0
            self.cancel_plan('UART_REOPEN')
            time.sleep(UART_STARTUP_DELAY_S)
            self.write_line('run')
            self.last_cmd_send_time = 0.0
            self.get_logger().info('Đã mở UART')
        except (serial.SerialException, OSError) as exc:
            self.serial_conn = None
            self.get_logger().error(f'Không mở được {self.serial_port}: {exc}')

    def close_serial(self) -> None:
        if self.serial_conn is not None:
            try:
                if self.serial_conn.is_open:
                    self.serial_conn.close()
            except serial.SerialException:
                pass
        self.serial_conn = None

    def write_line(self, text: str) -> None:
        if self.serial_conn is None or not self.serial_conn.is_open:
            return
        try:
            self.serial_conn.write((text.strip() + '\n').encode('utf-8'))
        except (serial.SerialException, OSError) as exc:
            self.get_logger().error(f'Lỗi ghi UART: {exc}')
            self.close_serial()

    def read_serial(self) -> None:
        if self.serial_conn is None or not self.serial_conn.is_open:
            return

        try:
            waiting = self.serial_conn.in_waiting
            if waiting > 0:
                self.rx_buffer += self.serial_conn.read(waiting).decode(
                    'utf-8', errors='ignore'
                )
        except (serial.SerialException, OSError) as exc:
            self.get_logger().error(f'Lỗi đọc UART: {exc}')
            self.close_serial()
            return

        while '\n' in self.rx_buffer:
            line, self.rx_buffer = self.rx_buffer.split('\n', 1)
            self.parse_uart_line(line.strip())

        if len(self.rx_buffer) > 2048:
            self.rx_buffer = self.rx_buffer[-512:]

    def parse_uart_line(self, line: str) -> None:
        parts = [part.strip() for part in line.split(',')]
        if len(parts) < 3 or parts[0].upper() != 'ENC':
            return

        try:
            left = int(parts[1])
            right = int(parts[2])
        except ValueError:
            return

        if not (
            0 <= left <= ENCODER_MAX_COUNT
            and 0 <= right <= ENCODER_MAX_COUNT
        ):
            return

        now = time.monotonic()
        self.encoder_left = left
        self.encoder_right = right

        new_delta_left = 0
        new_delta_right = 0
        if self.previous_encoder_left is not None:
            new_delta_left = self.encoder_delta_with_wrap(
                left,
                self.previous_encoder_left,
            )
        if self.previous_encoder_right is not None:
            new_delta_right = self.encoder_delta_with_wrap(
                right,
                self.previous_encoder_right,
            )

        self.previous_encoder_left = left
        self.previous_encoder_right = right
        self.delta_left = new_delta_left
        self.delta_right = new_delta_right
        self.encoder_last_time = now

        # Đây là số xung chênh lệch theo thời gian của từng bánh.
        # Không lấy encoder_left - encoder_right.
        self.encoder_accumulated_left += float(new_delta_left)
        self.encoder_accumulated_right += float(new_delta_right)

        if self.plan_active:
            if self.plan_left_direction != 0:
                self.plan_left_progress_pulses += float(new_delta_left)
            if self.plan_right_direction != 0:
                self.plan_right_progress_pulses += float(new_delta_right)

    def send_motor_pwm(self, left: int, right: int, force: bool = False) -> None:
        left = int(self.clamp(left, -255, 255))
        right = int(self.clamp(right, -255, 255))
        now = time.monotonic()

        same_command = (
            left == self.last_sent_left
            and right == self.last_sent_right
        )
        if (
            not force
            and same_command
            and now - self.last_cmd_send_time < self.cmd_heartbeat_s
        ):
            return

        self.write_line(f'CMD,{left},{right}')
        self.last_sent_left = left
        self.last_sent_right = right
        self.last_cmd_send_time = now

    # ============================================================
    # Pulse plan
    # ============================================================
    def cancel_plan(self, reason: str) -> None:
        self.plan_active = False
        self.plan_state = reason
        self.plan_start_time = 0.0

        self.plan_linear_mm = 0.0
        self.plan_yaw_rad = 0.0
        self.plan_depth_start_mm = 0.0
        self.plan_error_x_start = 0.0
        self.plan_yaw_sign = 0

        self.plan_left_distance_mm = 0.0
        self.plan_right_distance_mm = 0.0
        self.plan_left_direction = 0
        self.plan_right_direction = 0
        self.plan_left_target_pulses = 0.0
        self.plan_right_target_pulses = 0.0
        self.plan_left_progress_pulses = 0.0
        self.plan_right_progress_pulses = 0.0

    def distance_to_stop_zone(self) -> float:
        """Trả về quãng đường thân xe cần đi; dương tiến, âm lùi."""
        if self.depth_mm > self.stop_distance_far_mm:
            return min(
                self.depth_mm - self.stop_distance_far_mm,
                self.max_linear_segment_mm,
            )
        if self.depth_mm < self.stop_distance_near_mm:
            return -min(
                self.stop_distance_near_mm - self.depth_mm,
                self.max_linear_segment_mm,
            )
        return 0.0

    def yaw_error_rad(self) -> float:
        """Đổi error_x pixel thành góc yaw bằng mô hình pinhole."""
        if abs(self.error_x) <= self.x_deadzone_px:
            return 0.0

        raw_angle = math.atan2(self.error_x, self.camera_fx_px)
        desired = self.yaw_sign * raw_angle
        return self.clamp(
            desired,
            -self.max_yaw_segment_rad,
            self.max_yaw_segment_rad,
        )

    def create_plan(self, now: float) -> bool:
        """Đo depth/error_x rồi tạo mục tiêu xung riêng cho hai bánh."""
        linear_mm = self.distance_to_stop_zone()
        yaw_rad = self.yaw_error_rad()

        if abs(linear_mm) < self.min_linear_plan_mm:
            linear_mm = 0.0
        if abs(yaw_rad) < self.min_yaw_plan_rad:
            yaw_rad = 0.0

        # Target lệch nhiều: quay từng đoạn trước để hạn chế mất target.
        if abs(self.error_x) >= self.yaw_only_threshold_px:
            linear_mm = 0.0

        # Khi vừa đi vừa cua, giới hạn cung quay để hai bánh không đổi chiều.
        turn_arc_mm = 0.5 * self.wheel_track_mm * yaw_rad
        if linear_mm != 0.0 and turn_arc_mm != 0.0:
            max_arc = (
                abs(linear_mm)
                * self.max_turn_arc_to_linear_ratio
            )
            turn_arc_mm = self.clamp(turn_arc_mm, -max_arc, max_arc)
            yaw_rad = 2.0 * turn_arc_mm / self.wheel_track_mm

        # Mô hình differential drive:
        # d = (sL + sR)/2
        # yaw = (sR - sL)/track
        left_distance_mm = linear_mm - turn_arc_mm
        right_distance_mm = linear_mm + turn_arc_mm

        left_direction = self.sign(left_distance_mm)
        right_direction = self.sign(right_distance_mm)

        left_target = abs(left_distance_mm) / self.mm_per_pulse
        right_target = abs(right_distance_mm) / self.mm_per_pulse

        # Trừ dự phòng vọt lố trước khi thực thi.
        if left_target > 0.0:
            left_target = max(0.0, left_target - self.pulse_brake_margin)
        if right_target > 0.0:
            right_target = max(0.0, right_target - self.pulse_brake_margin)

        if left_target < self.pulse_min_target:
            left_target = 0.0
            left_direction = 0
            left_distance_mm = 0.0
        if right_target < self.pulse_min_target:
            right_target = 0.0
            right_direction = 0
            right_distance_mm = 0.0

        if left_target <= 0.0 and right_target <= 0.0:
            self.plan_state = 'DISTANCE_AND_YAW_OK'
            return False

        self.plan_active = True
        self.plan_start_time = now
        self.plan_state = 'PULSE_PLAN_ACTIVE'
        self.plan_linear_mm = linear_mm
        self.plan_yaw_rad = yaw_rad
        self.plan_depth_start_mm = self.depth_mm
        self.plan_error_x_start = self.error_x
        self.plan_yaw_sign = self.sign(yaw_rad)

        self.plan_left_distance_mm = left_distance_mm
        self.plan_right_distance_mm = right_distance_mm
        self.plan_left_direction = left_direction
        self.plan_right_direction = right_direction
        self.plan_left_target_pulses = left_target
        self.plan_right_target_pulses = right_target
        self.plan_left_progress_pulses = 0.0
        self.plan_right_progress_pulses = 0.0

        self.get_logger().info(
            'NEW_PULSE_PLAN | '
            f'depth={self.depth_mm:.0f}mm, errX={self.error_x:.0f}px, '
            f'linear={linear_mm:.1f}mm, '
            f'yaw={math.degrees(yaw_rad):.2f}deg, '
            f'L={left_distance_mm:.1f}mm/'
            f'{left_target:.1f}pulse, '
            f'R={right_distance_mm:.1f}mm/'
            f'{right_target:.1f}pulse'
        )
        return True

    def should_replan(self, now: float) -> Optional[str]:
        if not self.plan_active:
            return None

        elapsed = now - self.plan_start_time
        if elapsed >= self.plan_timeout_s:
            return 'PLAN_TIMEOUT'

        if elapsed < self.plan_min_run_before_replan_s:
            return None

        # Nếu đã vào vùng khoảng cách cho phép thì không tiếp tục đoạn tịnh tiến cũ.
        if self.plan_linear_mm != 0.0:
            if (
                self.stop_distance_near_mm
                <= self.depth_mm
                <= self.stop_distance_far_mm
            ):
                return 'REPLAN_DISTANCE_REACHED'

        if self.plan_yaw_rad != 0.0:
            current_yaw = self.yaw_error_rad()
            current_sign = self.sign(current_yaw)

            if (
                self.replan_when_x_centered
                and abs(self.error_x) <= self.x_deadzone_px
            ):
                return 'REPLAN_X_CENTERED'

            if (
                self.replan_on_yaw_sign_change
                and current_sign != 0
                and self.plan_yaw_sign != 0
                and current_sign != self.plan_yaw_sign
            ):
                return 'REPLAN_YAW_SIGN_CHANGED'

        return None

    def wheel_pulse_command(
        self,
        target_pulses: float,
        progress_pulses: float,
        direction: int,
    ) -> tuple[int, float, bool]:
        remaining = max(0.0, target_pulses - progress_pulses)

        if (
            direction == 0
            or target_pulses <= 0.0
            or remaining <= self.pulse_stop_margin
        ):
            return 0, remaining, True

        if remaining >= self.pulse_slowdown_remaining:
            magnitude = float(self.pulse_cruise_cmd)
        else:
            ratio = self.clamp(
                remaining / self.pulse_slowdown_remaining,
                0.0,
                1.0,
            )
            magnitude = (
                self.pulse_approach_cmd
                + ratio
                * (self.pulse_cruise_cmd - self.pulse_approach_cmd)
            )

        return int(round(direction * magnitude)), remaining, False

    def plan_motor_targets(
        self,
        now: float,
    ) -> tuple[int, int, str, float, float]:
        reason = self.should_replan(now)
        if reason is not None:
            self.cancel_plan(reason)
            self.plan_settle_until = now + self.plan_settle_s
            return 0, 0, reason, 0.0, 0.0

        left_cmd, left_remaining, left_done = self.wheel_pulse_command(
            self.plan_left_target_pulses,
            self.plan_left_progress_pulses,
            self.plan_left_direction,
        )
        right_cmd, right_remaining, right_done = self.wheel_pulse_command(
            self.plan_right_target_pulses,
            self.plan_right_progress_pulses,
            self.plan_right_direction,
        )

        if left_done and right_done:
            self.cancel_plan('PLAN_COMPLETE')
            self.plan_settle_until = now + self.plan_settle_s
            return 0, 0, 'PLAN_COMPLETE', left_remaining, right_remaining

        left_cmd, right_cmd = self.limit_wheel_cmd_difference(
            left_cmd,
            right_cmd,
        )
        left_cmd = int(self.clamp(
            left_cmd,
            -self.max_output_pwm,
            self.max_output_pwm,
        ))
        right_cmd = int(self.clamp(
            right_cmd,
            -self.max_output_pwm,
            self.max_output_pwm,
        ))

        return (
            left_cmd,
            right_cmd,
            'RUN_PULSE_PLAN',
            left_remaining,
            right_remaining,
        )

    # ============================================================
    # Ramp
    # ============================================================
    def ramp_one_wheel(self, current: int, target: int) -> int:
        # Bánh đã đủ xung hoặc dừng an toàn: về 0 ngay ở RPi.
        if target == 0:
            return 0

        # Nếu đổi chiều, giảm về 0 trước.
        if current * target < 0:
            return self.approach(current, 0, self.max_pwm_step)

        return self.approach(current, target, self.max_pwm_step)

    def apply_ramp(self, left_target: int, right_target: int) -> tuple[int, int]:
        if left_target == 0 and right_target == 0:
            self.current_left_pwm = 0
            self.current_right_pwm = 0
            return 0, 0

        self.current_left_pwm = self.ramp_one_wheel(
            self.current_left_pwm,
            left_target,
        )
        self.current_right_pwm = self.ramp_one_wheel(
            self.current_right_pwm,
            right_target,
        )

        self.current_left_pwm, self.current_right_pwm = (
            self.limit_wheel_cmd_difference(
                self.current_left_pwm,
                self.current_right_pwm,
            )
        )
        return self.current_left_pwm, self.current_right_pwm

    # ============================================================
    # Main loop
    # ============================================================
    def control_loop(self) -> None:
        now = time.monotonic()
        self.read_serial()

        if self.serial_conn is None or not self.serial_conn.is_open:
            self.current_left_pwm = 0
            self.current_right_pwm = 0
            self.cancel_plan('UART_DISCONNECTED')
            if now - self.last_reconnect_attempt >= UART_RECONNECT_INTERVAL_S:
                self.open_serial()
            return

        valid, safety_state = self.tracking_valid(now)
        if not valid:
            self.cancel_plan(safety_state)
            self.current_left_pwm = 0
            self.current_right_pwm = 0
            self.send_motor_pwm(0, 0)
            self.log_status(now, safety_state, 0, 0, 0.0, 0.0)
            return

        if not self.encoder_valid(now):
            self.cancel_plan('STOP_ENCODER_TIMEOUT')
            self.current_left_pwm = 0
            self.current_right_pwm = 0
            self.send_motor_pwm(0, 0)
            self.log_status(now, 'STOP_ENCODER_TIMEOUT', 0, 0, 0.0, 0.0)
            return

        if now < self.plan_settle_until:
            self.current_left_pwm = 0
            self.current_right_pwm = 0
            self.send_motor_pwm(0, 0)
            self.log_status(now, 'PLAN_SETTLE', 0, 0, 0.0, 0.0)
            return

        if not self.plan_active:
            created = self.create_plan(now)
            if not created:
                self.current_left_pwm = 0
                self.current_right_pwm = 0
                self.send_motor_pwm(0, 0)
                self.log_status(
                    now,
                    self.plan_state,
                    0,
                    0,
                    0.0,
                    0.0,
                )
                return

        (
            left_target,
            right_target,
            state,
            left_remaining,
            right_remaining,
        ) = self.plan_motor_targets(now)

        left_output, right_output = self.apply_ramp(
            left_target,
            right_target,
        )
        self.send_motor_pwm(left_output, right_output)
        self.log_status(
            now,
            state,
            left_target,
            right_target,
            left_remaining,
            right_remaining,
        )

    def log_status(
        self,
        now: float,
        state: str,
        left_target: int,
        right_target: int,
        left_remaining: float,
        right_remaining: float,
    ) -> None:
        if now - self.last_log_time < LOG_INTERVAL_S:
            return

        self.get_logger().info(
            f'{state} | mode={self.mode_id}, locked={int(self.locked)}, '
            f'lost={self.lost_frames}, depth={self.depth_mm:.0f}mm, '
            f'errX={self.error_x:.0f}px, fx={self.camera_fx_px:.1f}px, '
            f'encL={self.encoder_left}, encR={self.encoder_right}, '
            f'dL={self.delta_left}, dR={self.delta_right}, '
            f'planL={self.plan_left_progress_pulses:.1f}/'
            f'{self.plan_left_target_pulses:.1f}, '
            f'planR={self.plan_right_progress_pulses:.1f}/'
            f'{self.plan_right_target_pulses:.1f}, '
            f'remL={left_remaining:.1f}, remR={right_remaining:.1f}, '
            f'TARGET={left_target},{right_target}, '
            f'CMD={self.current_left_pwm},{self.current_right_pwm}'
        )
        self.last_log_time = now

    def destroy_node(self) -> bool:
        try:
            self.cancel_plan('NODE_DESTROY')
            self.send_motor_pwm(0, 0, force=True)
            time.sleep(UART_STOP_DELAY_S)
        finally:
            self.close_serial()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EncoderPulseTargetControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
