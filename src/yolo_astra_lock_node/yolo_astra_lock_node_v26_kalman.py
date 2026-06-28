#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Astra Mini S - YOLO + ByteTrack + Kalman + HARD Depth Lock v26 PERSON FIRST

Mục tiêu:
- Dựa trên ver22 ổn định, không lấy ver24 làm nền.
- Ưu tiên bám NGƯỜI; object mode chỉ là phần phụ.
- Kéo ROI nếu trúng/gần người YOLO detect được -> PERSON MODE.
- Nếu ROI hoàn toàn không trúng người -> OBJECT MODE bằng template matching.
- PERSON HARD-LOCK:
  + không tự đổi sang người khác khi chưa bấm c
  + nếu người khác đi ngang thì giữ/mất target, robot dừng an toàn
  + lưu đặc trưng người lúc lock ban đầu để reacquire người cũ
  + nếu người cũ quay lại, có thể nhận lại dù ByteTrack đổi ID
- Chống nhảy target bằng:
  + track_id lock nhưng không tin ID tuyệt đối
  + ReID nhẹ bằng HSV histogram + template gốc + depth + size
  + depth hard gate
  + color histogram gate
  + Kalman vận tốc không đổi cho tâm mục tiêu
  + mỗi kết quả YOLO chỉ được hiệu chỉnh một lần
  + position/motion prediction gate
  + size gate
  + lost/reacquire logic
  + template fallback ngắn hạn

Phím:
- Kéo chuột trái: chọn target
- c: clear target
- r: xoay hình 0/90/180/270
- q hoặc ESC: thoát
"""

import time
import threading
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YoloAstraLockNode(Node):
    def __init__(self):
        super().__init__('yolo_astra_lock_node')

        # ============================================================
        # SETTING CONFIG - chỉnh chủ yếu ở khu này
        # ============================================================

        # ROS topics của Orbbec Astra Mini S
        self.color_topic = '/camera/color/image_raw'
        self.depth_topic = '/camera/depth/image_raw'

        # Bật/tắt mode
        self.enable_person_mode = True
        self.enable_object_mode = True
        self.target_mode = 'none'  # none / person / object

        # Hiệu năng trên Raspberry Pi 5
        # track/display chạy nhanh hơn, YOLO chạy thấp hơn để đỡ tụt FPS
        self.target_track_fps = 15.0
        self.target_display_fps = 15.0
        self.target_yolo_fps = 4.0
        self.max_detection_age_sec = 0.45  # quá thời gian này không dùng detection cũ nữa

        # Xoay hình: 0/90/180/270
        self.rotate_mode = 180
        self._rotate_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        # YOLO person
        self.yolo_model_path = 'yolov8n.pt'
        self.yolo_imgsz = 224
        self.yolo_conf = 0.30
        self.yolo_classes = [0]          # COCO person = 0
        self.yolo_tracker = 'bytetrack.yaml'
        self.min_person_conf = 0.25
        # Ưu tiên bắt người: chỉ cần ROI chạm nhẹ hoặc gần bbox người là chọn PERSON.
        self.person_select_iou_min = 0.02
        self.person_select_expand_px = 35
        self.person_select_center_bonus = 520.0

        # PERSON FIRST / HARD LOCK CONFIG
        # Khi đã lock người, không cho chọn target mới bằng chuột nếu chưa bấm c.
        # Khi mất người, vẫn giữ bộ nhớ target để chờ người cũ quay lại.
        self.person_clear_only_by_key = True
        self.person_publish_locked_zero_when_lost = True
        self.person_no_auto_switch = True
        self.person_allow_reid_after_lost = 8
        self.person_reid_confirm_frames = 2
        self.person_reid_score_threshold = 0.58
        self.person_reid_color_min = 0.42
        self.person_reid_template_min = 0.12
        self.person_reid_depth_gate_mm = 900
        self.person_reid_size_min = 0.25
        self.person_reid_size_max = 3.80

        # ============================================================
        # LOCK / ANTI-SWITCH CONFIG
        # ============================================================

        # Person scoring weights. Tổng không cần đúng 1 tuyệt đối, code sẽ clamp về 0..1.
        self.person_w_id = 0.32
        self.person_w_iou = 0.12
        self.person_w_pos = 0.18
        self.person_w_depth = 0.22
        self.person_w_color = 0.12
        self.person_w_size = 0.04

        # Object/template scoring weights
        self.obj_w_template = 0.30
        self.obj_w_pos = 0.22
        self.obj_w_depth = 0.24
        self.obj_w_color = 0.19
        self.obj_w_size = 0.05

        # Ngưỡng nhận target
        self.person_final_score_threshold = 0.50
        self.object_final_score_threshold = 0.48
        self.template_person_threshold = 0.47

        # Nếu candidate khác ID, chỉ cho đổi sau khi mất target đủ lâu + điểm rất cao + xác nhận nhiều frame
        self.strict_id_lock = True
        # HARD LOCK v8: không tự đổi sang ID khác sớm. Nếu ByteTrack đổi ID,
        # ưu tiên giữ target cũ/template trước, chỉ đổi khi mất rất lâu.
        self.allow_switch_after_lost = 18
        self.switch_score_threshold = 0.70
        self.switch_confirm_frames = 3
        self.other_id_penalty = 0.45

        # Gate chống ByteTrack kéo nhầm cùng ID sang người/vật đi ngang.
        self.same_id_depth_gate_mm = 420
        self.same_id_pos_min = 0.10
        self.same_id_color_min_when_depth_bad = 0.52
        self.max_center_jump_px = 115
        self.front_occlusion_size_ratio_max = 1.55

        # Gate cứng cho PERSON
        # Giảm depth gate nếu bạn muốn chống người đi ngang mạnh hơn.
        self.depth_gate_mm = 700
        self.person_hard_depth_gate_mm = 480
        self.person_hard_color_min = 0.42
        self.person_hard_pos_min = 0.22
        self.person_hard_size_min = 0.30
        self.person_hard_size_max = 3.00

        # Gate cứng cho OBJECT
        self.object_hard_depth_gate_mm = 380          # giảm so với person để chặt hơn với object nhỏ
        self.object_hard_color_min = 0.48             # tăng: object nhỏ cần match màu tốt hơn
        self.object_hard_pos_min = 0.30               # tăng: không cho nhảy xa
        self.object_hard_size_min = 0.40              # candidate không được nhỏ hơn 40% area ban đầu
        self.object_hard_size_max = 2.20              # candidate không được lớn hơn 220% area ban đầu

        # ---- OBJECT INITIAL LOCK GATES (so với ROI/vật BAN ĐẦU, không chỉ frame trước) ----
        # Đây là vũ khí chính chống lại việc nhảy sang tóc/vùng tối lớn hơn
        self.object_max_area_ratio_from_initial = 2.0   # candidate không được > 2x diện tích ban đầu
        self.object_min_area_ratio_from_initial = 0.35  # candidate không được < 35% diện tích ban đầu
        self.object_max_center_jump_from_initial_px = 180  # tâm candidate không được quá xa tâm lock ban đầu (dùng khi lost lâu)
        self.object_search_margin_base = 55             # search ROI base margin (nhỏ hơn để không match vùng lớn xa)
        self.object_search_margin_max = 90              # giới hạn mở rộng search khi lost (không mở quá rộng)
        self.object_template_update_min_score = 0.72    # chỉ update template khi rất chắc (tăng từ 0.62)
        self.object_template_update_min_color = 0.60   # tăng từ 0.50
        self.object_template_update_min_depth = 0.65   # tăng từ 0.55
        self.object_final_score_threshold = 0.52        # tăng từ 0.48 để reject candidate mơ hồ
        # Số frame lost trước khi mở rộng search (để không quá sớm match vùng xa)
        self.object_hold_before_expand = 8

        # Chống vật/người đi ngang phía trước camera cướp lock
        self.occlusion_hold_enable = True
        self.occlusion_front_mm = 260       # candidate gần hơn target cũ > 260mm thì nghi che khuất
        self.occlusion_color_min = 0.52     # nếu màu/vị trí/size đáng nghi thì giữ target cũ

        # Search ROI + reacquire
        self.search_margin = 60
        self.max_reacquire_dist_px = 175
        self.reacquire_extra_margin_per_lost = 4
        self.reacquire_max_extra_margin = 150
        self.max_lost = 90

        # Motion prediction
        self.use_motion_prediction = True

        # Kalman trạng thái [x, y, vx, vy], vận tốc tính theo pixel/giây.
        # YOLO/template là phép đo; Kalman predict chạy mỗi frame tracking.
        self.use_kalman_center = True
        self.kalman_measurement_noise_px = 11.0
        self.kalman_template_noise_scale = 1.45
        self.kalman_accel_noise_px_s2 = 320.0
        self.kalman_max_speed_px_s = 900.0
        self.kalman_dt_min = 1.0 / 60.0
        self.kalman_dt_max = 0.20

        # Cho phép dự đoán rất ngắn khi đúng một vài frame không có phép đo.
        # 3 frame ở 15 Hz xấp xỉ 0.2 s, sau đó vẫn LOST và dừng an toàn.
        self.person_kalman_coast_frames = 3
        self.person_kalman_coast_max_age_sec = 0.22

        # EMA vận tốc cũ được giữ làm fallback khi tắt Kalman.
        self.velocity_smooth_alpha = 0.35
        self.max_velocity_px = 75

        # Template matching
        self.min_roi_size = 20
        self.match_threshold = 0.26
        self.match_threshold_lost = 0.20
        self.top_k_candidates = 6
        self.multi_scale_factors = [0.85, 0.95, 1.0, 1.08, 1.18]
        self.person_template_fallback_max_lost = 22

        # Template update rất nhẹ để không bị drift sang vật khác
        self.template_update_alpha = 0.035
        self.template_update_min_score = 0.62
        self.template_update_min_color = 0.50
        self.template_update_min_depth = 0.55

        # Depth
        self.depth_radius = 8
        self.depth_roi_margin = 8
        self.min_valid_depth = 1
        self.depth_smooth_alpha = 0.25
        self.depth_hold_frames = 12

        # Control output
        self.center_deadzone_px = 25
        # Tâm đã được Kalman lọc nên EMA sai số dùng alpha cao hơn để giảm trễ.
        self.error_smooth_alpha = 0.55

        # Display
        self.panel_w = 420
        self.show_search_area = True
        self.show_camera_center = True
        self.window_name = 'Astra Mini S - v26 KALMAN PERSON FIRST HARD LOCK'

        self.color_person = (255, 0, 255)
        self.color_object = (0, 0, 255)
        self.color_search = (0, 140, 255)
        self.color_center = (0, 255, 255)
        self.color_line = (0, 255, 0)
        self.color_panel_border = (0, 255, 0)
        self.color_panel_bg = (25, 25, 25)
        self.color_text = (0, 255, 255)
        self.color_gray_box = (90, 90, 90)
        self.color_bad = (0, 0, 255)
        self.color_good = (0, 255, 0)

        # Nếu YOLO lỗi thì object mode vẫn chạy
        self.disable_yolo_on_error = True

        # ============================================================
        # INTERNAL STATE
        # ============================================================

        self.bridge = CvBridge()
        self.model = None

        self.latest_color_msg = None
        self.latest_depth_msg = None
        self.pending_select_box = None

        self._frame_front = None
        self._frame_back = None
        self.lock = threading.Lock()
        self._stop_event = threading.Event()

        # YOLO detection state
        self.last_yolo_time = 0.0
        self.last_yolo_det_time = 0.0
        self.person_detections = []
        # Mỗi lần YOLO thực sự chạy sẽ tăng sequence. update_person chỉ tiêu thụ
        # mỗi sequence một lần, tránh dùng lặp bbox cũ ở nhiều frame tracking.
        self.yolo_result_seq = 0
        self.last_consumed_yolo_seq = -1
        self.yolo_fps = 0.0
        self._yolo_count = 0
        self._yolo_fps_time = time.time()
        self.yolo_error_count = 0

        # Target state
        self.target_locked = False
        self.target_box = None
        self.target_center = None
        self.last_good_box = None
        self.last_good_center = None
        self.selected_track_id = None

        self.target_template = None
        self.target_template_color = None
        self.target_hist = None
        self.target_area = 0

        # ---- OBJECT INITIAL LOCK STATE ----
        # Luu thong tin tai thoi diem lock ban dau de so sanh voi candidate
        # khong chi so voi frame truoc. Day la "neo" cung chong drift sang vung khac.
        self.lock_initial_box = None
        self.lock_initial_area = 0
        self.lock_initial_center = None
        self.lock_initial_depth_mm = 0
        self.lock_initial_template = None   # KHONG bao gio update sau lock
        self.lock_initial_hist = None       # KHONG bao gio update sau lock
        self.lock_initial_track_id = None

        # Điểm ReID hiển thị trên panel. Đây chỉ là ReID nhẹ, không phải nhận diện mặt.
        self.last_init_color_score = 0.0
        self.last_init_template_score = 0.0
        self.last_reid_score = 0.0

        self.last_depth_mm = 0
        self.last_valid_depth_mm = 0
        self.depth_smooth = 0.0
        self.depth_invalid_count = 0

        self.lock_depth_ref_mm = 0
        self.lock_color_ref_ready = False

        # Mouse state
        self.dragging = False
        self.drag_start = None
        self.drag_end = None

        # FPS
        self.last_process_time = 0.0
        self.track_fps = 0.0
        self._fps_count = 0
        self._fps_time = time.time()

        # Lost/reacquire
        self.lost_count = 0
        self.pending_switch_track_id = None
        self.pending_switch_count = 0

        # Motion
        self.prev_center = None
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        # Kalman center state
        self.kalman = None
        self.kalman_initialized = False
        self.kalman_predicted_center = None
        self.kalman_last_predict_time = None
        self.kalman_last_measurement_time = 0.0
        self.kalman_coast_count = 0

        # Output info
        self.last_error_x = 0
        self.last_error_y = 0
        self.error_x_smooth = 0.0
        self.error_y_smooth = 0.0
        self.last_direction = 'NO TARGET'
        self.last_status = 'NO TARGET'
        self.last_reject_reason = 'none'

        self.last_final_score = 0.0
        self.last_match_score = 0.0
        self.last_id_score = 0.0
        self.last_iou_score = 0.0
        self.last_pos_score = 0.0
        self.last_depth_score = 0.0
        self.last_color_score = 0.0
        self.last_size_score = 0.0

        # ============================================================
        # INIT YOLO
        # ============================================================

        if self.enable_person_mode:
            if YOLO is None:
                self.get_logger().error('Chua cai ultralytics: pip install ultralytics')
                self.enable_person_mode = False
            else:
                self.get_logger().info('Dang load YOLO model...')
                self.model = YOLO(self.yolo_model_path)
                self.get_logger().info('Load YOLO model xong')

        # ============================================================
        # ROS SUBSCRIBERS
        # ============================================================

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(Image, self.color_topic, self.color_callback, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self.depth_callback, sensor_qos)

        # Publisher cho node control đọc dữ liệu bám target
        # /target_tracking = [locked, error_x, error_y, depth_mm, lost_count, mode_id]
        self.tracking_pub = self.create_publisher(
            Float32MultiArray,
            '/target_tracking',
            10
        )

        # ============================================================
        # GUI + THREAD
        # ============================================================

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.worker_thread = threading.Thread(target=self.process_loop, daemon=True)
        self.worker_thread.start()

        self.display_timer = self.create_timer(1.0 / self.target_display_fps, self.display_callback)

        self.get_logger().info('Astra Mini S YOLO Depth Lock v25 PERSON FIRST started')
        self.get_logger().info(f'Color topic: {self.color_topic}')
        self.get_logger().info(f'Depth topic: {self.depth_topic}')

    # ============================================================
    # ROS CALLBACKS
    # ============================================================

    def color_callback(self, msg):
        with self.lock:
            self.latest_color_msg = msg

    def depth_callback(self, msg):
        with self.lock:
            self.latest_depth_msg = msg

    # ============================================================
    # MOUSE
    # ============================================================

    def mouse_callback(self, event, x, y, flags, param):
        with self.lock:
            frame = self._frame_front

        if frame is None:
            return

        img_w = frame.shape[1] - self.panel_w
        img_h = frame.shape[0]
        if x >= img_w:
            return

        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.target_locked:
                self.get_logger().warn('Dang hard-lock target. Bam c de clear roi moi chon target khac.')
                return
            self.dragging = True
            self.drag_start = (x, y)
            self.drag_end = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.drag_end = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.drag_end = (x, y)
            if self.drag_start is None:
                return

            x1 = min(self.drag_start[0], self.drag_end[0])
            y1 = min(self.drag_start[1], self.drag_end[1])
            x2 = max(self.drag_start[0], self.drag_end[0])
            y2 = max(self.drag_start[1], self.drag_end[1])

            if (x2 - x1) < self.min_roi_size or (y2 - y1) < self.min_roi_size:
                self.get_logger().warn('ROI qua nho, bo qua')
                return

            with self.lock:
                self.pending_select_box = (x1, y1, x2, y2)

            self.get_logger().info(f'Select ROI: ({x1},{y1})-({x2},{y2})')

    # ============================================================
    # BASIC UTILS
    # ============================================================

    def rotate_image(self, img):
        code = self._rotate_map.get(self.rotate_mode)
        return cv2.rotate(img, code) if code is not None else img

    def change_rotate_mode(self):
        cycle = {0: 90, 90: 180, 180: 270, 270: 0}
        self.rotate_mode = cycle.get(self.rotate_mode, 0)
        self.get_logger().info(f'rotate_mode = {self.rotate_mode}')

    @staticmethod
    def clamp_box(box, w, h):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return (x1, y1, x2, y2)

    @staticmethod
    def box_center(box):
        return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    @staticmethod
    def center_dist_px(a, b):
        """Khoảng cách pixel giữa 2 tâm.

        Bản v8 bị thiếu hàm này nên process_loop lỗi liên tục sau khi LOCK.
        Nếu một trong hai tâm chưa có, trả 0 để không làm crash frame xử lý.
        """
        if a is None or b is None:
            return 0.0
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return (dx * dx + dy * dy) ** 0.5

    @staticmethod
    def box_area(box):
        if box is None:
            return 0
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    @staticmethod
    def box_area_ratio(new_box, old_box):
        """Tỉ lệ diện tích bbox mới so với bbox cũ.

        Dùng cho hard-lock size gate. Trả 1.0 khi thiếu dữ liệu để
        không làm chết process_loop và không phạt sai khi chưa có old_box.
        """
        if new_box is None or old_box is None:
            return 1.0
        old_a = max(0, old_box[2] - old_box[0]) * max(0, old_box[3] - old_box[1])
        new_a = max(0, new_box[2] - new_box[0]) * max(0, new_box[3] - new_box[1])
        if old_a <= 0 or new_a <= 0:
            return 1.0
        return float(new_a) / float(old_a)

    @staticmethod
    def iou(a, b):
        if a is None or b is None:
            return 0.0
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
        ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
        union = aa + ab - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def point_in_box(pt, box):
        return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

    def crop_box(self, frame, box):
        if frame is None or box is None:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.clamp_box(box, w, h)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _old_ref(self):
        old_box = self.last_good_box if self.last_good_box is not None else self.target_box
        old_center = self.last_good_center if self.last_good_center is not None else self.target_center
        return old_box, old_center

    def make_search_roi(self, box, w, h):
        # OBJECT MODE: search ROI nho hon, mo rong cham hon de tranh match vung xa
        if self.target_mode == 'object':
            base_margin = self.object_search_margin_base
            max_extra = self.object_search_margin_max
            # Chi mo rong sau khi da lost du lau (object_hold_before_expand)
            extra_frames = max(0, self.lost_count - self.object_hold_before_expand)
            extra = min(extra_frames * self.reacquire_extra_margin_per_lost, max_extra)
        else:
            extra = 0
            if self.lost_count > 0:
                extra = min(self.lost_count * self.reacquire_extra_margin_per_lost,
                            self.reacquire_max_extra_margin)
            base_margin = self.search_margin

        m = base_margin + extra
        return self.clamp_box((box[0] - m, box[1] - m, box[2] + m, box[3] + m), w, h)

    # ============================================================
    # DEPTH
    # ============================================================

    def get_depth_mm(self, depth_img, box):
        if depth_img is None or box is None:
            return 0

        h, w = depth_img.shape[:2]
        x1, y1, x2, y2 = self.clamp_box(box, w, h)
        cx, cy = self.box_center((x1, y1, x2, y2))

        # Ưu tiên vùng nhỏ quanh tâm bbox vì ít dính nền hơn
        r = self.depth_radius
        roi = depth_img[max(0, cy - r):min(h, cy + r + 1),
                        max(0, cx - r):min(w, cx + r + 1)]
        valid = roi[roi >= self.min_valid_depth]
        if valid.size > 0:
            return self.robust_median(valid)

        # Fallback: vùng trong bbox, bỏ viền để giảm dính background
        m = self.depth_roi_margin
        bx1 = min(max(x1 + m, 0), w - 1)
        by1 = min(max(y1 + m, 0), h - 1)
        bx2 = max(min(x2 - m, w), bx1 + 1)
        by2 = max(min(y2 - m, h), by1 + 1)
        roi2 = depth_img[by1:by2, bx1:bx2]
        valid2 = roi2[roi2 >= self.min_valid_depth]
        if valid2.size > 0:
            return self.robust_median(valid2)

        return 0

    @staticmethod
    def robust_median(values):
        if values.size == 0:
            return 0
        if values.size < 8:
            return int(np.median(values))
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        filtered = values[(values >= low) & (values <= high)]
        if filtered.size > 0:
            return int(np.median(filtered))
        return int(np.median(values))

    def update_depth_smooth(self, depth_mm):
        if depth_mm > 0:
            self.last_valid_depth_mm = depth_mm
            self.depth_invalid_count = 0
            if self.depth_smooth <= 0:
                self.depth_smooth = float(depth_mm)
            else:
                a = self.depth_smooth_alpha
                self.depth_smooth = (1 - a) * self.depth_smooth + a * depth_mm
            return int(self.depth_smooth)

        self.depth_invalid_count += 1
        if self.depth_invalid_count <= self.depth_hold_frames:
            return self.last_valid_depth_mm
        return 0

    # ============================================================
    # COLOR / TEMPLATE
    # ============================================================

    @staticmethod
    def preprocess_template(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def calc_hsv_hist(self, crop):
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if h < 4 or w < 4:
            return None

        # Mask bỏ viền để histogram ít dính nền
        my = max(1, h // 10)
        mx = max(1, w // 10)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[my:h - my, mx:w - mx] = 255

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], mask, [36, 40], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist

    def color_score(self, crop):
        if self.target_hist is None:
            return 0.5
        hist = self.calc_hsv_hist(crop)
        if hist is None:
            return 0.0
        raw = cv2.compareHist(self.target_hist, hist, cv2.HISTCMP_CORREL)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def color_score_vs_initial(self, crop):
        """So sanh mau crop voi histogram TAI THOI DIEM LOCK BAN DAU (khong bao gio update)."""
        if self.lock_initial_hist is None:
            return 0.5
        hist = self.calc_hsv_hist(crop)
        if hist is None:
            return 0.0
        raw = cv2.compareHist(self.lock_initial_hist, hist, cv2.HISTCMP_CORREL)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    def template_score_vs_initial(self, frame, box):
        """So sanh template crop voi template TAI THOI DIEM LOCK BAN DAU (khong bao gio update)."""
        if self.lock_initial_template is None:
            return self.template_score_for_box(frame, box)  # fallback sang current template
        crop = self.crop_box(frame, box)
        if crop is None:
            return 0.0
        th, tw = self.lock_initial_template.shape[:2]
        if th < self.min_roi_size or tw < self.min_roi_size:
            return 0.0
        try:
            resized = cv2.resize(crop, (tw, th))
            gray = self.preprocess_template(resized)
            result = cv2.matchTemplate(gray, self.lock_initial_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(result)
            return max(0.0, min(1.0, float(val)))
        except Exception:
            return 0.0

    def template_score_for_box(self, frame, box):
        if self.target_template is None:
            return 0.5
        crop = self.crop_box(frame, box)
        if crop is None:
            return 0.0
        th, tw = self.target_template.shape[:2]
        if th < self.min_roi_size or tw < self.min_roi_size:
            return 0.0
        try:
            resized = cv2.resize(crop, (tw, th))
            gray = self.preprocess_template(resized)
            result = cv2.matchTemplate(gray, self.target_template, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(result)
            return max(0.0, min(1.0, float(val)))
        except Exception:
            return 0.0

    # ============================================================
    # SCORE HELPERS
    # ============================================================

    def reset_motion_filter(self, center=None):
        """Khởi tạo lại bộ dự đoán chuyển động tại tâm vừa lock."""
        self.prev_center = center
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.kalman = None
        self.kalman_initialized = False
        self.kalman_predicted_center = center
        self.kalman_last_predict_time = None
        self.kalman_last_measurement_time = 0.0
        self.kalman_coast_count = 0

        if not self.use_kalman_center or center is None:
            return

        kf = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)
        kf.measurementMatrix = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        kf.transitionMatrix = np.eye(4, dtype=np.float32)
        sigma = float(self.kalman_measurement_noise_px)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * (sigma * sigma)
        kf.processNoiseCov = np.eye(4, dtype=np.float32)
        kf.errorCovPost = np.diag([25.0, 25.0, 400.0, 400.0]).astype(np.float32)

        state = np.array(
            [[float(center[0])], [float(center[1])], [0.0], [0.0]],
            dtype=np.float32,
        )
        kf.statePost = state.copy()
        kf.statePre = state.copy()

        self.kalman = kf
        self.kalman_initialized = True
        now = time.monotonic()
        self.kalman_last_predict_time = now
        self.kalman_last_measurement_time = now

    def predict_motion_filter(self, now=None):
        """Chạy bước predict đúng một lần cho mỗi frame tracking."""
        if not self.use_motion_prediction:
            return self.last_good_center

        if not self.use_kalman_center or not self.kalman_initialized or self.kalman is None:
            if self.last_good_center is None:
                return None
            self.kalman_predicted_center = (
                int(self.last_good_center[0] + self.velocity_x),
                int(self.last_good_center[1] + self.velocity_y),
            )
            return self.kalman_predicted_center

        if now is None:
            now = time.monotonic()
        if self.kalman_last_predict_time is None:
            dt = 1.0 / max(self.target_track_fps, 1.0)
        else:
            dt = now - self.kalman_last_predict_time
        dt = max(self.kalman_dt_min, min(self.kalman_dt_max, dt))
        self.kalman_last_predict_time = now

        self.kalman.transitionMatrix = np.array(
            [[1.0, 0.0, dt, 0.0],
             [0.0, 1.0, 0.0, dt],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        q = float(self.kalman_accel_noise_px_s2) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        self.kalman.processNoiseCov = np.array(
            [[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
             [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
             [dt3 / 2.0, 0.0, dt2, 0.0],
             [0.0, dt3 / 2.0, 0.0, dt2]],
            dtype=np.float32,
        ) * q

        prediction = self.kalman.predict()
        max_speed = float(self.kalman_max_speed_px_s)
        prediction[2, 0] = np.clip(prediction[2, 0], -max_speed, max_speed)
        prediction[3, 0] = np.clip(prediction[3, 0], -max_speed, max_speed)

        # Nếu frame này không có correct, statePost vẫn phải tiến theo dự đoán
        # để frame sau tiếp tục từ vị trí mới, không dự đoán lặp một bước cũ.
        self.kalman.statePre = prediction.copy()
        self.kalman.statePost = prediction.copy()
        self.kalman.errorCovPost = self.kalman.errorCovPre.copy()

        self.kalman_predicted_center = (
            int(round(float(prediction[0, 0]))),
            int(round(float(prediction[1, 0]))),
        )
        return self.kalman_predicted_center

    def update_velocity(self, new_center, measurement_noise_scale=1.0):
        """Cập nhật EMA fallback và correct Kalman bằng một phép đo tâm mới."""
        if new_center is None:
            return self.predicted_center()

        if self.prev_center is None:
            self.prev_center = new_center
            self.velocity_x = 0.0
            self.velocity_y = 0.0
        else:
            dx = new_center[0] - self.prev_center[0]
            dy = new_center[1] - self.prev_center[1]
            dx = max(-self.max_velocity_px, min(self.max_velocity_px, dx))
            dy = max(-self.max_velocity_px, min(self.max_velocity_px, dy))

            a = self.velocity_smooth_alpha
            self.velocity_x = (1 - a) * self.velocity_x + a * dx
            self.velocity_y = (1 - a) * self.velocity_y + a * dy
            self.prev_center = new_center

        if not self.use_kalman_center:
            self.kalman_coast_count = 0
            return (int(new_center[0]), int(new_center[1]))

        if not self.kalman_initialized or self.kalman is None:
            self.reset_motion_filter(new_center)
            return (int(new_center[0]), int(new_center[1]))

        sigma = float(self.kalman_measurement_noise_px) * max(0.5, float(measurement_noise_scale))
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * (sigma * sigma)
        measurement = np.array(
            [[float(new_center[0])], [float(new_center[1])]],
            dtype=np.float32,
        )
        corrected = self.kalman.correct(measurement)

        max_speed = float(self.kalman_max_speed_px_s)
        corrected[2, 0] = np.clip(corrected[2, 0], -max_speed, max_speed)
        corrected[3, 0] = np.clip(corrected[3, 0], -max_speed, max_speed)
        self.kalman.statePost = corrected.copy()

        filtered = (
            int(round(float(corrected[0, 0]))),
            int(round(float(corrected[1, 0]))),
        )
        self.kalman_predicted_center = filtered
        self.kalman_last_measurement_time = time.monotonic()
        self.kalman_coast_count = 0
        return filtered

    def predicted_center(self):
        if not self.use_motion_prediction:
            return self.last_good_center
        if self.use_kalman_center and self.kalman_initialized:
            return self.kalman_predicted_center or self.last_good_center
        if self.last_good_center is None:
            return None
        return (int(self.last_good_center[0] + self.velocity_x),
                int(self.last_good_center[1] + self.velocity_y))

    def apply_person_kalman_coast(self, frame):
        """Giữ tracking tối đa vài frame bằng predict, không nhận phép đo giả."""
        if not self.use_kalman_center or not self.kalman_initialized:
            return False
        if self.kalman_coast_count >= self.person_kalman_coast_frames:
            return False
        if self.kalman_last_measurement_time <= 0.0:
            return False
        if time.monotonic() - self.kalman_last_measurement_time > self.person_kalman_coast_max_age_sec:
            return False

        predicted = self.predicted_center()
        base_box = self.last_good_box if self.last_good_box is not None else self.target_box
        if predicted is None or base_box is None:
            return False

        h, w = frame.shape[:2]
        base_center = self.box_center(base_box)
        dx = int(predicted[0] - base_center[0])
        dy = int(predicted[1] - base_center[1])
        shifted = self.clamp_box(
            (base_box[0] + dx, base_box[1] + dy,
             base_box[2] + dx, base_box[3] + dy),
            w,
            h,
        )
        if self.box_area(shifted) <= 0:
            return False

        self.target_box = shifted
        self.target_center = predicted
        self.kalman_coast_count += 1
        self.last_status = (
            f'PERSON KALMAN COAST '
            f'{self.kalman_coast_count}/{self.person_kalman_coast_frames}'
        )
        self.last_reject_reason = 'temporary measurement miss'
        return True

    def position_score(self, new_center, old_center, max_dist):
        if new_center is None or old_center is None:
            return 0.5
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]
        dist = math.sqrt(dx * dx + dy * dy)
        return max(0.0, min(1.0, 1.0 - dist / max(max_dist, 1)))

    def depth_score(self, new_depth, old_depth):
        if new_depth <= 0 or old_depth <= 0:
            return 0.5
        diff = abs(new_depth - old_depth)
        return max(0.0, min(1.0, 1.0 - diff / max(self.depth_gate_mm, 1)))

    def size_ratio(self, new_box, old_box):
        old_a = self.box_area(old_box) if old_box else 0
        new_a = self.box_area(new_box) if new_box else 0
        if old_a <= 0 or new_a <= 0:
            return 1.0
        return new_a / float(old_a)

    def size_score(self, new_box, old_box, r_min, r_max):
        ratio = self.size_ratio(new_box, old_box)
        if ratio < r_min or ratio > r_max:
            return 0.0
        # score cao nhất khi ratio gần 1
        return max(0.0, min(1.0, 1.0 - abs(1.0 - ratio)))

    def _set_score_pack(self, final, id_s=0.0, iou_s=0.0, pos_s=0.0, dep_s=0.0,
                        col_s=0.0, siz_s=0.0, tpl_s=0.0):
        self.last_final_score = float(final)
        self.last_id_score = float(id_s)
        self.last_iou_score = float(iou_s)
        self.last_pos_score = float(pos_s)
        self.last_depth_score = float(dep_s)
        self.last_color_score = float(col_s)
        self.last_size_score = float(siz_s)
        self.last_match_score = float(tpl_s)

    # ============================================================
    # YOLO PERSON
    # ============================================================

    def run_yolo_person(self, frame):
        if not self.enable_person_mode or self.model is None:
            return False

        now = time.time()
        if now - self.last_yolo_time < 1.0 / max(self.target_yolo_fps, 1.0):
            return False

        self.last_yolo_time = now
        try:
            results = self.model.track(
                frame,
                persist=True,
                tracker=self.yolo_tracker,
                classes=self.yolo_classes,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                verbose=False,
            )
            self.person_detections = self.extract_persons(results)
            self.last_yolo_det_time = time.time()
            self.yolo_result_seq += 1

            self._yolo_count += 1
            dt = now - self._yolo_fps_time
            if dt >= 1.0:
                self.yolo_fps = self._yolo_count / dt
                self._yolo_count = 0
                self._yolo_fps_time = now
            return True

        except Exception as e:
            self.yolo_error_count += 1
            self.get_logger().warn(f'YOLO loi: {e}')
            if self.disable_yolo_on_error:
                self.get_logger().warn('Tat PERSON MODE do YOLO loi, OBJECT MODE van chay')
                self.enable_person_mode = False
            return False

    def detections_are_fresh(self):
        if not self.person_detections:
            return False
        return (time.time() - self.last_yolo_det_time) <= self.max_detection_age_sec

    def extract_persons(self, results):
        dets = []
        if not results or len(results) == 0:
            return dets
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return dets

        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(conf))
        ids = boxes.id.cpu().numpy() if boxes.id is not None else None

        for i in range(len(conf)):
            if int(cls[i]) != 0:
                continue
            x1, y1, x2, y2 = map(int, xyxy[i])
            if x2 <= x1 or y2 <= y1:
                continue
            b = (x1, y1, x2, y2)
            tid = int(ids[i]) if ids is not None else None
            dets.append({
                'box': b,
                'center': self.box_center(b),
                'area': self.box_area(b),
                'conf': float(conf[i]),
                'track_id': tid,
            })
        return dets

    # ============================================================
    # LOCK TARGET FROM ROI
    # ============================================================

    def lock_target_from_roi(self, frame, depth, roi_box):
        h, w = frame.shape[:2]
        roi_box = self.clamp_box(roi_box, w, h)

        # Nếu đã có person detection trong ROI thì lock person
        if self.enable_person_mode and self.detections_are_fresh():
            det = self.find_person_from_roi(roi_box)
            if det is not None:
                self.lock_target(frame, depth, det['box'], 'person', det['center'], det['track_id'])
                return

        # Nếu không trúng person thì lock object bằng đúng ROI
        if self.enable_object_mode:
            self.lock_target(frame, depth, roi_box, 'object', self.box_center(roi_box), None)

    def find_person_from_roi(self, roi_box):
        """Chọn person tốt nhất từ ROI.

        Bản v26 ưu tiên người hơn ver22: ngoài IoU/tâm nằm trong ROI,
        còn xét ROI đã mở rộng một chút để người dùng chỉ cần khoanh gần người.
        """
        rc = self.box_center(roi_box)
        ex = self.person_select_expand_px
        # expanded_roi sẽ được clamp sau khi biết bbox detection tương đối trong frame,
        # nên chỉ cần tạo dạng mở rộng ở đây.
        expanded_roi = (roi_box[0] - ex, roi_box[1] - ex,
                        roi_box[2] + ex, roi_box[3] + ex)

        best = None
        best_score = -1e9

        for det in self.person_detections:
            box = det['box']
            center = det['center']
            iou_s = self.iou(roi_box, box)
            iou_exp = self.iou(expanded_roi, box)
            center_inside = self.point_in_box(center, roi_box)
            center_inside_exp = self.point_in_box(center, expanded_roi)
            roi_center_inside_person = self.point_in_box(rc, box)

            if (iou_s < self.person_select_iou_min and
                iou_exp <= 0.0 and
                not center_inside and
                not center_inside_exp and
                not roi_center_inside_person):
                continue

            dx = center[0] - rc[0]
            dy = center[1] - rc[1]
            dist = math.sqrt(dx * dx + dy * dy)
            score = (
                iou_s * 1300.0 +
                iou_exp * 650.0 -
                dist +
                (self.person_select_center_bonus if center_inside else 0.0) +
                (260.0 if center_inside_exp else 0.0) +
                (360.0 if roi_center_inside_person else 0.0) +
                det['conf'] * 80.0
            )
            if score > best_score:
                best_score = score
                best = det

        return best

    def lock_target(self, frame, depth, box, mode, center=None, track_id=None):
        crop = self.crop_box(frame, box)
        if crop is None:
            self.get_logger().warn('Crop rong, khong lock')
            return

        self.target_template_color = crop.copy()
        self.target_template = self.preprocess_template(crop)
        self.target_hist = self.calc_hsv_hist(crop)
        self.lock_color_ref_ready = self.target_hist is not None
        self.target_area = self.box_area(box)

        dm = self.get_depth_mm(depth, box)
        self.last_depth_mm = dm
        self.last_valid_depth_mm = dm
        self.lock_depth_ref_mm = dm
        self.depth_smooth = float(dm) if dm > 0 else 0.0
        self.depth_invalid_count = 0

        # ---- Luu INITIAL LOCK STATE cho ca PERSON va OBJECT ----
        # Cac bien nay KHONG bao gio update sau khi lock.
        # PERSON dùng để ReID nhẹ khi mất người/ByteTrack đổi ID.
        # OBJECT dùng để chống drift sang vùng khác lớn hơn/tương tự màu.
        self.lock_initial_box = box
        self.lock_initial_area = self.target_area
        self.lock_initial_center = center if center is not None else self.box_center(box)
        self.lock_initial_depth_mm = dm
        self.lock_initial_template = self.target_template.copy()
        self.lock_initial_hist = self.target_hist.copy() if self.target_hist is not None else None
        self.lock_initial_track_id = track_id
        self.last_init_color_score = 1.0
        self.last_init_template_score = 1.0
        self.last_reid_score = 1.0

        self.target_mode = mode
        self.target_locked = True
        self.target_box = box
        self.target_center = center if center is not None else self.box_center(box)
        self.last_good_box = box
        self.last_good_center = self.target_center
        self.selected_track_id = track_id
        self.lost_count = 0
        self.pending_switch_track_id = None
        self.pending_switch_count = 0

        self.reset_motion_filter(self.target_center)
        if mode == 'person':
            # ROI vừa lock từ chính kết quả YOLO hiện tại, không correct lại
            # cùng bbox thêm một lần ngay trong process_frame này.
            self.last_consumed_yolo_seq = self.yolo_result_seq

        self.error_x_smooth = 0.0
        self.error_y_smooth = 0.0
        self.last_status = f'LOCK {mode.upper()}'
        self.last_reject_reason = 'none'
        self._set_score_pack(1.0, id_s=1.0 if track_id is not None else 0.0,
                             iou_s=1.0, pos_s=1.0, dep_s=1.0, col_s=1.0, siz_s=1.0, tpl_s=1.0)

        self.get_logger().info(f'LOCK {mode.upper()}: id={track_id}, depth={dm}mm, box={box}')

    # ============================================================
    # UPDATE TARGET
    # ============================================================

    def update_target(self, frame, depth):
        if not self.target_locked:
            return

        if self.target_mode == 'person':
            self.update_person(frame, depth)
        elif self.target_mode == 'object':
            self.update_object(frame, depth)

    def update_person(self, frame, depth):
        fresh = self.detections_are_fresh()
        has_new_yolo_result = self.yolo_result_seq != self.last_consumed_yolo_seq

        # Chỉ tiêu thụ mỗi kết quả YOLO đúng một lần. Trước đây cùng một bbox
        # được dùng lặp nhiều frame trong 0.45 s, làm vận tốc bị kéo về 0.
        if has_new_yolo_result:
            self.last_consumed_yolo_seq = self.yolo_result_seq

        # 1) Nếu có một kết quả YOLO MỚI và còn tươi, thử candidate tốt nhất.
        if has_new_yolo_result and fresh:
            best_det = None
            best_score = -1.0
            best_pack = None

            for det in self.person_detections:
                if det['conf'] < self.min_person_conf:
                    continue
                score, pack, reject = self.score_person_candidate(frame, depth, det)
                if score > best_score:
                    best_score = score
                    best_det = det
                    best_pack = pack
                    self.last_reject_reason = reject

            if best_det is not None and best_score >= self.person_final_score_threshold:
                best_id = best_det.get('track_id', None)
                other_id = (self.selected_track_id is not None and
                            best_id is not None and
                            best_id != self.selected_track_id)

                if other_id:
                    # v26: Không switch người khác. Chỉ nhận ID mới nếu score_person_candidate
                    # đã pass ReID nhẹ với đặc trưng người cũ và xác nhận vài frame.
                    if best_score < self.person_reid_score_threshold:
                        self.lost_count += 1
                        self.last_status = 'HOLD OLD PERSON'
                        self.last_reject_reason = f'reid score low {best_score:.2f}'
                        return

                    if self.pending_switch_track_id == best_id:
                        self.pending_switch_count += 1
                    else:
                        self.pending_switch_track_id = best_id
                        self.pending_switch_count = 1

                    if self.pending_switch_count < self.person_reid_confirm_frames:
                        self.lost_count += 1
                        self.last_status = 'CONFIRM OLD PERSON REID'
                        self.last_reject_reason = 'waiting reid confirm'
                        return

                    self.get_logger().info(f'REACQUIRE OLD PERSON with new ByteTrack id={best_id}, score={best_score:.2f}')

                self.apply_person_detection(frame, depth, best_det, 'TRACK PERSON', best_score, best_pack)
                self.pending_switch_track_id = None
                self.pending_switch_count = 0
                return

        # 2) Không có detection mới hoặc không candidate nào đạt: fallback bằng template trong vài frame.
        if self.lost_count < self.person_template_fallback_max_lost:
            if self.update_by_template(frame, depth, person_fallback=True):
                self.last_status = 'PERSON TEMPLATE HOLD'
                return

        # 3) Mất phép đo rất ngắn: dùng Kalman predict tối đa vài frame để
        # tránh locked bật/tắt chỉ vì một frame template/YOLO bị hụt.
        if self.apply_person_kalman_coast(frame):
            return

        # 4) Lost nhưng không tự chuyển sang người khác. Nếu cấu hình clear_only_by_key,
        # giữ bộ nhớ target và chờ người cũ quay lại; output sẽ locked=0 để robot dừng.
        self.lost_count += 1
        if self.person_clear_only_by_key:
            self.last_status = 'PERSON LOST - WAIT OLD TARGET'
            if self.lost_count == self.max_lost:
                self.get_logger().warn('PERSON lost qua lau nhung van giu memory. Bam c de clear neu muon chon lai.')
        else:
            self.last_status = 'PERSON LOST'
            if self.lost_count >= self.max_lost:
                self.get_logger().warn('Mat PERSON qua lau, clear target')
                self.clear_target()

    def person_reid_score_for_box(self, frame, depth, box):
        """ReID nhẹ cho người đã lock.

        Không dùng nhận diện khuôn mặt nặng. Điểm dựa trên đặc trưng đã lưu
        lúc khóa ban đầu: màu HSV, template grayscale, depth và size bbox.
        Mục tiêu là phân biệt người cũ với người khác đi ngang ở mức thực nghiệm.
        """
        crop = self.crop_box(frame, box)
        col_init = self.color_score_vs_initial(crop)
        tpl_init = self.template_score_vs_initial(frame, box)
        depth_new = self.get_depth_mm(depth, box)

        if depth_new > 0 and self.lock_initial_depth_mm > 0:
            depth_diff = abs(depth_new - self.lock_initial_depth_mm)
            depth_init = max(0.0, min(1.0, 1.0 - depth_diff / max(self.person_reid_depth_gate_mm, 1)))
        else:
            depth_init = 0.50

        size_init = self.size_score(
            box,
            self.lock_initial_box,
            self.person_reid_size_min,
            self.person_reid_size_max
        )

        # Template toàn thân thường biến thiên mạnh theo tư thế nên chỉ lấy trọng số vừa phải.
        reid_score = (
            0.38 * col_init +
            0.18 * tpl_init +
            0.28 * depth_init +
            0.16 * size_init
        )
        reid_score = max(0.0, min(1.0, reid_score))
        return reid_score, col_init, tpl_init, depth_init, size_init, depth_new

    def score_person_candidate(self, frame, depth, det):
        """Chấm điểm candidate PERSON với hard-lock và ReID nhẹ.

        Nguyên tắc v26:
        - Cùng track_id: vẫn phải qua gate depth/position/size như ver22.
        - Khác track_id: KHÔNG switch sớm. Chỉ nhận nếu đã lost đủ lâu và
          giống đặc trưng ban đầu của người cũ.
        - Người khác đi ngang/che camera: trả -1 để robot giữ/mất target,
          không cập nhật nhầm bbox.
        """
        box = det['box']
        center = det['center']
        old_box, old_center_raw = self._old_ref()
        pred = self.predicted_center()
        old_center = pred if pred is not None else old_center_raw

        same_id = (self.selected_track_id is not None and det['track_id'] == self.selected_track_id)
        other_id = (self.selected_track_id is not None and det['track_id'] is not None and
                    det['track_id'] != self.selected_track_id)

        id_s = 1.0 if same_id else 0.0
        iou_s = self.iou(old_box, box)
        pos_s = self.position_score(center, old_center, self.max_reacquire_dist_px)
        jump_px = self.center_dist_px(center, old_center)
        depth_new = self.get_depth_mm(depth, box)
        depth_old = self.last_valid_depth_mm
        dep_s = self.depth_score(depth_new, depth_old)
        col_s = self.color_score(self.crop_box(frame, box))
        siz_s = self.size_score(box, old_box, self.person_hard_size_min, self.person_hard_size_max)
        size_ratio = self.box_area_ratio(box, old_box)

        reid_s, col_init, tpl_init, dep_init_s, size_init_s, _ = self.person_reid_score_for_box(frame, depth, box)
        self.last_init_color_score = col_init
        self.last_init_template_score = tpl_init
        self.last_reid_score = reid_s

        pack = {
            'id_s': id_s,
            'iou_s': iou_s,
            'pos_s': pos_s,
            'dep_s': dep_s,
            'col_s': col_s,
            'siz_s': siz_s,
            'tpl_s': tpl_init,
        }

        # 1) Khác ID: không tự switch. Chỉ xem là người cũ khi đã lost đủ lâu
        # và ReID nhẹ đạt ngưỡng. Đây là lớp chống người khác đi ngang.
        if other_id and self.person_no_auto_switch:
            if self.lost_count < self.person_allow_reid_after_lost:
                return -1.0, pack, 'person lock: reject other id before lost'
            if reid_s < self.person_reid_score_threshold:
                return -1.0, pack, f'person reid low {reid_s:.2f}'
            if col_init < self.person_reid_color_min and tpl_init < self.person_reid_template_min:
                return -1.0, pack, f'person reid color/template low {col_init:.2f}/{tpl_init:.2f}'
            if depth_new > 0 and self.lock_initial_depth_mm > 0:
                depth_diff_init = abs(depth_new - self.lock_initial_depth_mm)
                if depth_diff_init > self.person_reid_depth_gate_mm:
                    return -1.0, pack, f'person reid depth {int(depth_diff_init)}mm'

        # 2) Nếu khác ID và chưa đủ điều kiện hard lock cũ thì reject như ver22.
        if other_id and self.strict_id_lock and self.lost_count < self.allow_switch_after_lost:
            return -1.0, pack, 'hardlock: reject other id'

        # 3) Candidate nhảy xa khỏi vị trí dự đoán: nghi ByteTrack kéo nhầm.
        # Khi đang lost lâu, nới rộng để cho phép reacquire người cũ ở vị trí mới.
        allowed_jump = self.max_center_jump_px + min(self.lost_count * 8, 120)
        if jump_px > allowed_jump and self.lost_count < self.allow_switch_after_lost:
            return -1.0, pack, f'hardlock: center jump {int(jump_px)}px'

        # 4) Size đổi quá mạnh: người đi ngang sát camera thường bbox to bất thường.
        if siz_s <= 0.0:
            return -1.0, pack, 'hardlock: size gate'

        # 5) Front occlusion: candidate gần hơn target cũ rõ rệt.
        if self.occlusion_hold_enable and depth_new > 0 and depth_old > 0:
            front_diff = depth_old - depth_new
            if front_diff > self.occlusion_front_mm:
                suspicious_front = (
                    col_s < self.occlusion_color_min or
                    pos_s < 0.62 or
                    size_ratio > self.front_occlusion_size_ratio_max or
                    iou_s < 0.20
                )
                if suspicious_front:
                    return -1.0, pack, 'hardlock: front occlusion hold'

        # 6) Depth gate áp dụng cả cùng ID.
        if depth_new > 0 and depth_old > 0:
            depth_diff = abs(depth_new - depth_old)
            if same_id:
                if depth_diff > self.same_id_depth_gate_mm:
                    if not (pos_s > 0.72 and col_s > self.same_id_color_min_when_depth_bad and iou_s > 0.18):
                        return -1.0, pack, f'hardlock: same id depth {int(depth_diff)}mm'
            else:
                if depth_diff > self.person_hard_depth_gate_mm and not (other_id and reid_s >= self.person_reid_score_threshold):
                    return -1.0, pack, 'hardlock: depth gate'

        # 7) Màu/vị trí tối thiểu. Với cùng ID mềm hơn, khác ID cứng hơn.
        if same_id:
            if pos_s < self.same_id_pos_min:
                return -1.0, pack, 'hardlock: same id pos gate'
            if col_s < 0.25 and dep_s < 0.55:
                return -1.0, pack, 'hardlock: same id color+depth gate'
        else:
            if col_s < self.person_hard_color_min and not (other_id and reid_s >= self.person_reid_score_threshold):
                return -1.0, pack, 'hardlock: color gate'
            if pos_s < self.person_hard_pos_min and self.lost_count < self.person_allow_reid_after_lost:
                return -1.0, pack, 'hardlock: position gate'

        final = (
            self.person_w_id * id_s +
            self.person_w_iou * iou_s +
            self.person_w_pos * pos_s +
            self.person_w_depth * dep_s +
            self.person_w_color * col_s +
            self.person_w_size * siz_s
        )

        if other_id:
            # Nếu khác ID nhưng pass ReID, dùng điểm ReID làm nền, không phạt kiểu ver22
            # để có thể bắt lại người cũ khi ByteTrack cấp ID mới.
            final = max(final, reid_s)
        elif self.selected_track_id is not None and det['track_id'] is None:
            final -= self.other_id_penalty * 0.5

        final = max(0.0, min(1.0, final))
        return final, pack, 'none'

    def apply_person_detection(self, frame, depth, det, status, score, pack):
        box = det['box']
        self.target_box = box
        measured_center = det['center']
        self.target_center = self.update_velocity(
            measured_center,
            measurement_noise_scale=1.0,
        )

        self.last_good_box = box
        self.last_good_center = self.target_center
        if det['track_id'] is not None:
            self.selected_track_id = det['track_id']

        self.lost_count = 0
        self.last_status = status
        self.last_reject_reason = 'none'
        self.update_depth_and_error(depth)

        if pack is not None:
            self._set_score_pack(score, **pack)

        self.update_template_if_good(frame, box, person_fallback=False)

    def update_object(self, frame, depth):
        if self.update_by_template(frame, depth, person_fallback=False):
            return

        self.lost_count += 1
        self.last_status = 'OBJECT LOST'
        if self.lost_count >= self.max_lost:
            self.get_logger().warn('Mat OBJECT qua lau, clear target')
            self.clear_target()

    def update_by_template(self, frame, depth, person_fallback=False):
        candidates = self.get_template_candidates(frame)
        if not candidates:
            return False

        old_box, old_center = self._old_ref()
        pred = self.predicted_center()
        if pred is not None:
            old_center = pred

        best_box = None
        best_score = -1.0
        best_pack = None
        best_reject = 'none'

        for cand in candidates:
            box = cand['box']
            tpl_s = cand['score']
            center = self.box_center(box)
            pos_s = self.position_score(center, old_center,
                                        self.max_reacquire_dist_px if person_fallback else self.object_search_margin_base + self.object_search_margin_max)
            depth_new = self.get_depth_mm(depth, box)
            depth_old = self.last_valid_depth_mm
            dep_s = self.depth_score(depth_new, depth_old)
            col_s = self.color_score(self.crop_box(frame, box))

            if person_fallback:
                siz_s = self.size_score(box, old_box, self.person_hard_size_min, self.person_hard_size_max)
                threshold = self.template_person_threshold
                w_tpl, w_pos, w_dep, w_col, w_siz = 0.34, 0.24, 0.20, 0.17, 0.05
                hard_depth = self.person_hard_depth_gate_mm
                hard_color = self.person_hard_color_min
                hard_pos = self.person_hard_pos_min
                mode_name = 'person template'
            else:
                # OBJECT MODE: so sanh voi old_box (frame truoc) va voi initial lock
                siz_s = self.size_score(box, old_box, self.object_hard_size_min, self.object_hard_size_max)
                threshold = self.object_final_score_threshold
                w_tpl, w_pos, w_dep, w_col, w_siz = (
                    self.obj_w_template, self.obj_w_pos, self.obj_w_depth,
                    self.obj_w_color, self.obj_w_size
                )
                hard_depth = self.object_hard_depth_gate_mm
                hard_color = self.object_hard_color_min
                hard_pos = self.object_hard_pos_min
                mode_name = 'object template'

            reject = 'none'
            size_ratio = self.box_area_ratio(box, old_box)
            jump_px = self.center_dist_px(center, old_center)

            # HARD LOCK v8 cho template: neu template match vao vat phia truoc/nhay xa
            # thi HOLD target cu, khong cap nhat box/template/depth.
            if depth_new > 0 and depth_old > 0 and depth_old - depth_new > self.occlusion_front_mm:
                if col_s < self.occlusion_color_min or pos_s < 0.62 or size_ratio > self.front_occlusion_size_ratio_max:
                    reject = f'{mode_name} front occlusion hold'
                    score = -1.0
                else:
                    score = None
            elif jump_px > self.max_center_jump_px + min(self.lost_count * 8, 90):
                reject = f'{mode_name} center jump'
                score = -1.0
            elif depth_new > 0 and depth_old > 0:
                if abs(depth_new - depth_old) > hard_depth:
                    reject = f'{mode_name} depth gate'
                    score = -1.0
                else:
                    score = None
            else:
                score = None

            if score is None:
                if col_s < hard_color:
                    reject = f'{mode_name} color gate'
                    score = -1.0
                elif pos_s < hard_pos:
                    reject = f'{mode_name} position gate'
                    score = -1.0
                elif siz_s <= 0.0:
                    reject = f'{mode_name} size gate'
                    score = -1.0
                else:
                    score = (w_tpl * tpl_s + w_pos * pos_s + w_dep * dep_s +
                             w_col * col_s + w_siz * siz_s)
                    score = max(0.0, min(1.0, score))

            # ================================================================
            # OBJECT MODE: INITIAL LOCK ANCHOR GATES
            # Sau khi pass cac gate tren, con phai so voi ROI BAN DAU de
            # chong truong hop "nhay sang toc/nguoi lon hon co cung mau".
            # ================================================================
            if score is not None and score >= 0.0 and not person_fallback:
                init_reject = self._object_initial_anchor_gate(frame, depth, box, center, depth_new)
                if init_reject is not None:
                    reject = init_reject
                    score = -1.0

            if score > best_score:
                best_score = score
                best_box = box
                best_reject = reject
                best_pack = {
                    'id_s': 0.0,
                    'iou_s': self.iou(old_box, box),
                    'pos_s': pos_s,
                    'dep_s': dep_s,
                    'col_s': col_s,
                    'siz_s': siz_s,
                    'tpl_s': tpl_s,
                }

        self.last_reject_reason = best_reject

        if best_box is None or best_score < threshold:
            return False

        self.target_box = best_box
        measured_center = self.box_center(best_box)
        noise_scale = self.kalman_template_noise_scale if person_fallback else 1.25
        self.target_center = self.update_velocity(
            measured_center,
            measurement_noise_scale=noise_scale,
        )
        self.last_good_box = best_box
        self.last_good_center = self.target_center
        self.lost_count = 0
        self.last_status = 'TRACK OBJECT' if not person_fallback else 'PERSON TEMPLATE HOLD'
        self.update_depth_and_error(depth)

        if best_pack is not None:
            self._set_score_pack(best_score, **best_pack)

        self.update_template_if_good(frame, best_box, person_fallback)
        return True

    def _object_initial_anchor_gate(self, frame, depth, box, center, depth_new):
        """
        Gate cung so voi ROI/vat BAN DAU khi lock.
        Tra ve reason string neu reject, None neu pass.

        Muc dich: chong truong hop candidate lon hon nhieu hoac nam xa
        vi tri ban dau (vi du toc/nguoi) danh bai dien thoai nho hon
        chi vi template match mau giong hon.
        """
        if self.lock_initial_area <= 0 or self.lock_initial_center is None:
            return None  # Khong co du lieu ban dau, bo qua gate nay

        cand_area = self.box_area(box)

        # 1) AREA GATE SO VOI BAN DAU: candidate khong duoc lon qua nhieu so voi ROI ban dau
        if self.lock_initial_area > 0 and cand_area > 0:
            area_ratio_from_initial = cand_area / float(self.lock_initial_area)
            if area_ratio_from_initial > self.object_max_area_ratio_from_initial:
                return f'obj_init: area too large ({area_ratio_from_initial:.1f}x initial)'
            if area_ratio_from_initial < self.object_min_area_ratio_from_initial:
                return f'obj_init: area too small ({area_ratio_from_initial:.2f}x initial)'

        # 2) DEPTH GATE SO VOI BAN DAU: depth khong duoc lech qua nhieu so voi depth ban dau
        if depth_new > 0 and self.lock_initial_depth_mm > 0:
            depth_diff_from_initial = abs(depth_new - self.lock_initial_depth_mm)
            # Gate nay mem hon (2x) so voi gate frame-to-frame vi object co the di xa hon
            if depth_diff_from_initial > self.object_hard_depth_gate_mm * 2.0:
                return f'obj_init: depth too far from initial ({int(depth_diff_from_initial)}mm)'

        # 3) COLOR GATE SO VOI BAN DAU: so mau voi histogram lock ban dau
        crop = self.crop_box(frame, box)
        if crop is not None:
            col_vs_initial = self.color_score_vs_initial(crop)
            if col_vs_initial < self.object_hard_color_min - 0.08:
                # Yeu cau mau gan voi mau ban dau (cho phep mem hon 0.08 de bu drift nhe)
                return f'obj_init: color mismatch vs initial ({col_vs_initial:.2f})'

        # 4) TEMPLATE GATE SO VOI BAN DAU: match voi template goc (khong bao gio update)
        tpl_vs_initial = self.template_score_vs_initial(frame, box)
        if tpl_vs_initial < self.match_threshold - 0.05:
            return f'obj_init: template mismatch vs initial ({tpl_vs_initial:.2f})'

        return None  # Pass tat ca gate

    def get_template_candidates(self, frame):
        cands = []
        if self.target_template is None or self.target_box is None:
            return cands

        h, w = frame.shape[:2]
        th, tw = self.target_template.shape[:2]
        if th < self.min_roi_size or tw < self.min_roi_size:
            return cands

        search_box = self.make_search_roi(self.target_box, w, h)
        sx1, sy1, sx2, sy2 = search_box
        search_img = frame[sy1:sy2, sx1:sx2]
        if search_img.size == 0:
            return cands

        search_gray = self.preprocess_template(search_img)
        sh, sw = search_gray.shape[:2]
        threshold = self.match_threshold_lost if self.lost_count > 0 else self.match_threshold

        for scale in self.multi_scale_factors:
            stw = max(self.min_roi_size, int(tw * scale))
            sth = max(self.min_roi_size, int(th * scale))
            if stw > sw or sth > sh:
                continue

            tpl = self.target_template if abs(scale - 1.0) < 1e-6 else cv2.resize(self.target_template, (stw, sth))
            result = cv2.matchTemplate(search_gray, tpl, cv2.TM_CCOEFF_NORMED)
            work = result.copy()

            for _ in range(self.top_k_candidates):
                _, max_val, _, max_loc = cv2.minMaxLoc(work)
                if max_val < threshold:
                    break

                mx, my = max_loc
                box = self.clamp_box((sx1 + mx, sy1 + my, sx1 + mx + stw, sy1 + my + sth), w, h)
                cands.append({'box': box, 'score': float(max_val)})

                # non-maximum suppression đơn giản trên map template
                rx1 = max(0, mx - stw // 2)
                ry1 = max(0, my - sth // 2)
                rx2 = min(work.shape[1] - 1, mx + stw // 2)
                ry2 = min(work.shape[0] - 1, my + sth // 2)
                work[ry1:ry2 + 1, rx1:rx2 + 1] = -1.0

        cands.sort(key=lambda x: x['score'], reverse=True)
        return cands[:self.top_k_candidates]

    def update_template_if_good(self, frame, box, person_fallback=False):
        # Cap nhat template rat de dat de tranh drift sang nguoi/vat khac.
        if self.target_template is None:
            return

        # Object mode dung nguong rieng (cao hon) duoc dinh nghia trong config
        if not person_fallback and self.target_mode == 'object':
            min_score = self.object_template_update_min_score
            min_color = self.object_template_update_min_color
            min_depth = self.object_template_update_min_depth
        else:
            min_score = self.template_update_min_score
            min_color = self.template_update_min_color
            min_depth = self.template_update_min_depth

        if self.last_match_score < min_score:
            return
        if self.last_color_score < min_color:
            return
        if self.last_depth_score < min_depth:
            return

        crop = self.crop_box(frame, box)
        if crop is None:
            return

        new_tpl = self.preprocess_template(crop)
        if new_tpl.shape != self.target_template.shape:
            return

        # Object mode: alpha nho hon (update rat cham) de tranh drift
        alpha = self.template_update_alpha if (person_fallback or self.target_mode != 'object') else self.template_update_alpha * 0.5
        self.target_template = cv2.addWeighted(self.target_template, 1 - alpha, new_tpl, alpha, 0)

        new_hist = self.calc_hsv_hist(crop)
        if new_hist is not None and self.target_hist is not None:
            self.target_hist = cv2.addWeighted(self.target_hist, 1 - alpha, new_hist, alpha, 0)
        # NOTE: lock_initial_template va lock_initial_hist KHONG BAO GIO bi cap nhat o day.

    def update_depth_and_error(self, depth):
        if not self.target_locked or self.target_box is None:
            return
        raw = self.get_depth_mm(depth, self.target_box)
        self.last_depth_mm = self.update_depth_smooth(raw)
        self.last_good_box = self.target_box
        self.last_good_center = self.target_center

    def update_error_direction(self, frame):
        if not self.target_locked or self.target_center is None:
            self.last_direction = 'NO TARGET'
            self.last_error_x = 0
            self.last_error_y = 0
            return

        # Khi PERSON đang lost, giữ memory bên trong nhưng không xuất sai error cũ.
        if self.target_mode == 'person' and self.lost_count > 0:
            self.last_direction = 'PERSON LOST'
            self.last_error_x = 0
            self.last_error_y = 0
            self.error_x_smooth = 0.0
            self.error_y_smooth = 0.0
            return

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        tx, ty = self.target_center

        a = self.error_smooth_alpha
        self.error_x_smooth = (1 - a) * self.error_x_smooth + a * (tx - cx)
        self.error_y_smooth = (1 - a) * self.error_y_smooth + a * (ty - cy)

        self.last_error_x = int(self.error_x_smooth)
        self.last_error_y = int(self.error_y_smooth)

        if self.last_error_x > self.center_deadzone_px:
            self.last_direction = 'RIGHT'
        elif self.last_error_x < -self.center_deadzone_px:
            self.last_direction = 'LEFT'
        else:
            self.last_direction = 'CENTER'

    def publish_tracking(self):
        """
        Publish dữ liệu tracking cho control.py.

        msg.data = [locked, error_x, error_y, depth_mm, lost_count, mode_id]
        mode_id: 0 none, 1 person, 2 object

        v26: Nếu PERSON đang lost, vẫn giữ internal memory để reacquire,
        nhưng publish locked=0, error=0, depth=0 để control node dừng robot.
        """
        msg = Float32MultiArray()

        if self.target_mode == 'person':
            mode_id = 1.0
        elif self.target_mode == 'object':
            mode_id = 2.0
        else:
            mode_id = 0.0

        locked = 1.0 if self.target_locked else 0.0
        err_x = float(self.last_error_x)
        err_y = float(self.last_error_y)
        depth_mm = float(self.last_depth_mm)

        if (self.target_mode == 'person' and
            self.person_publish_locked_zero_when_lost and
            self.lost_count > 0):
            locked = 0.0
            err_x = 0.0
            err_y = 0.0
            depth_mm = 0.0

        msg.data = [locked, err_x, err_y, depth_mm, float(self.lost_count), mode_id]
        self.tracking_pub.publish(msg)

    def clear_target(self):
        self.target_mode = 'none'
        self.target_locked = False
        self.target_box = None
        self.target_center = None
        self.last_good_box = None
        self.last_good_center = None
        self.selected_track_id = None

        self.target_template = None
        self.target_template_color = None
        self.target_hist = None
        self.target_area = 0

        # Reset initial lock state
        self.lock_initial_box = None
        self.lock_initial_area = 0
        self.lock_initial_center = None
        self.lock_initial_depth_mm = 0
        self.lock_initial_template = None
        self.lock_initial_hist = None
        self.lock_initial_track_id = None
        self.last_init_color_score = 0.0
        self.last_init_template_score = 0.0
        self.last_reid_score = 0.0

        self.last_depth_mm = 0
        self.last_valid_depth_mm = 0
        self.lock_depth_ref_mm = 0
        self.depth_smooth = 0.0
        self.depth_invalid_count = 0

        self.lost_count = 0
        self.pending_switch_track_id = None
        self.pending_switch_count = 0

        self.reset_motion_filter(None)
        self.last_consumed_yolo_seq = self.yolo_result_seq

        self.last_error_x = 0
        self.last_error_y = 0
        self.error_x_smooth = 0.0
        self.error_y_smooth = 0.0
        self.last_direction = 'NO TARGET'
        self.last_status = 'NO TARGET'
        self.last_reject_reason = 'none'
        self._set_score_pack(0.0)

    # ============================================================
    # PROCESS LOOP
    # ============================================================

    def process_loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            period = 1.0 / max(self.target_track_fps, 1.0)
            if now - self.last_process_time < period:
                self._stop_event.wait(0.002)
                continue
            self.last_process_time = now

            with self.lock:
                color_msg = self.latest_color_msg
                depth_msg = self.latest_depth_msg
                select_box = self.pending_select_box
                self.pending_select_box = None

            if color_msg is None:
                self._stop_event.wait(0.005)
                continue

            try:
                frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
                frame = self.rotate_image(frame)

                depth = None
                if depth_msg is not None:
                    try:
                        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
                        depth = self.rotate_image(depth)
                    except Exception:
                        depth = None

                output = self.process_frame(frame, depth, select_box)

                with self.lock:
                    self._frame_back = output
                    self._frame_front, self._frame_back = self._frame_back, self._frame_front

            except Exception as e:
                self.get_logger().error(f'Loi process_loop: {e}')
                self._stop_event.wait(0.02)

    def process_frame(self, frame, depth, select_box):
        self._fps_count += 1
        now = time.time()
        dt = now - self._fps_time
        if dt >= 1.0:
            self.track_fps = self._fps_count / dt
            self._fps_count = 0
            self._fps_time = now

        # YOLO có thể chạy thấp FPS; phần tracking vẫn chạy bằng template/depth ở giữa các lần YOLO.
        self.run_yolo_person(frame)

        if select_box is not None:
            self.lock_target_from_roi(frame, depth, select_box)

        if self.target_locked:
            # Predict trước khi chấm gate/correct để mọi candidate được so với
            # vị trí kỳ vọng ở đúng thời điểm frame hiện tại.
            self.predict_motion_filter(time.monotonic())
            self.update_target(frame, depth)

        self.update_error_direction(frame)
        self.publish_tracking()
        return self.make_canvas(frame)

    # ============================================================
    # DRAW
    # ============================================================

    def put(self, img, text, pos, scale=0.45, color=None, thick=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color if color is not None else self.color_text, thick)

    def make_canvas(self, frame):
        h, w = frame.shape[:2]
        canvas = np.empty((h, w + self.panel_w, 3), dtype=np.uint8)
        canvas[:, 0:w] = frame
        self.draw_camera(canvas, w, h)
        self.draw_panel(canvas, w, h)
        return canvas

    def draw_camera(self, canvas, iw, ih):
        frame = canvas[:, 0:iw]
        cx, cy = iw // 2, ih // 2

        if self.show_camera_center:
            cv2.circle(frame, (cx, cy), 5, self.color_center, -1)
            cv2.line(frame, (cx - 12, cy), (cx + 12, cy), self.color_center, 1)
            cv2.line(frame, (cx, cy - 12), (cx, cy + 12), self.color_center, 1)

        # Draw all person detections mờ
        if self.enable_person_mode and self.detections_are_fresh():
            for det in self.person_detections:
                b = det['box']
                tid = det['track_id']
                label = f'id {tid}' if tid is not None else 'person'
                cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), self.color_gray_box, 1)
                self.put(frame, label, (b[0], max(b[1] - 6, 14)), 0.42, self.color_gray_box)

        if self.target_locked and self.target_box is not None:
            if self.show_search_area:
                s = self.make_search_roi(self.target_box, iw, ih)
                cv2.rectangle(frame, (s[0], s[1]), (s[2], s[3]), self.color_search, 1)
                self.put(frame, 'SEARCH', (s[0], max(s[1] - 6, 16)), 0.45, self.color_search)

            b = self.target_box
            c = self.target_center
            color = self.color_person if self.target_mode == 'person' else self.color_object
            label = 'TARGET PERSON' if self.target_mode == 'person' else 'TARGET OBJECT'

            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), color, 2)
            cv2.circle(frame, c, 5, color, -1)
            cv2.line(frame, (cx, cy), c, self.color_line, 2)
            self.put(frame, label, (b[0], max(b[1] - 8, 18)), 0.55, color, 2)

        # Draw ROI đang kéo
        if self.dragging and self.drag_start is not None and self.drag_end is not None:
            d = (min(self.drag_start[0], self.drag_end[0]),
                 min(self.drag_start[1], self.drag_end[1]),
                 max(self.drag_start[0], self.drag_end[0]),
                 max(self.drag_start[1], self.drag_end[1]))
            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), self.color_center, 2)
            self.put(frame, 'SELECT', (d[0], max(d[1] - 8, 18)), 0.55, self.color_center, 2)

    def draw_panel(self, canvas, iw, ih):
        x0 = iw
        canvas[:, x0:x0 + self.panel_w] = self.color_panel_bg
        cv2.rectangle(canvas, (x0, 0), (x0 + self.panel_w - 1, ih - 1), self.color_panel_border, 1)

        y = 28
        status_color = self.color_good if self.target_locked else self.color_bad
        status_text = self.last_status if self.target_locked else 'NO TARGET'

        self.put(canvas, 'v26 KALMAN PERSON LOCK', (x0 + 15, y), 0.60, (255, 255, 255), 2)
        y += 34
        self.put(canvas, f'STATUS: {status_text}', (x0 + 15, y), 0.52, status_color, 2)
        y += 28

        dep_txt = f'depth: {self.last_depth_mm} mm' if self.last_depth_mm > 0 else 'depth: invalid'
        fresh_txt = 'fresh' if self.detections_are_fresh() else 'stale/no det'

        lines = [
            f'mode: {self.target_mode.upper()}',
            f'track FPS: {self.track_fps:.1f}',
            f'YOLO FPS: {self.yolo_fps:.1f}',
            f'YOLO det: {fresh_txt}',
            f'rotate: {self.rotate_mode}',
            f'track id: {self.selected_track_id}',
            f'final: {self.last_final_score:.2f}',
            f'id: {self.last_id_score:.2f}',
            f'iou: {self.last_iou_score:.2f}',
            f'pos: {self.last_pos_score:.2f}',
            f'depth_s: {self.last_depth_score:.2f}',
            f'color: {self.last_color_score:.2f}',
            f'size: {self.last_size_score:.2f}',
            f'template: {self.last_match_score:.2f}',
            f'reid: {self.last_reid_score:.2f}',
            f'init col/tpl: {self.last_init_color_score:.2f}/{self.last_init_template_score:.2f}',
            f'reject: {self.last_reject_reason}',
            f'reid confirm: {self.pending_switch_count}/{self.person_reid_confirm_frames}',
            f'lost: {self.lost_count}/{self.max_lost}',
            f'err_x: {self.last_error_x} px',
            f'err_y: {self.last_error_y} px',
            dep_txt,
            f'dir: {self.last_direction}',
            f'clear only by c: {self.person_clear_only_by_key}',
        ]

        for t in lines:
            self.put(canvas, t, (x0 + 15, y), 0.41)
            y += 18

        y += 8
        help_lines = [
            'Drag ROI: lock person/object',
            'c: clear target',
            'r: rotate 0/90/180/270',
            'q/ESC: quit',
            'PERSON FIRST + NO AUTO SWITCH',
            'Anti-switch + depth gate ON',
        ]
        for t in help_lines:
            self.put(canvas, t, (x0 + 15, y), 0.35, (220, 220, 220))
            y += 17

    # ============================================================
    # DISPLAY
    # ============================================================

    def display_callback(self):
        with self.lock:
            frame = self._frame_front
        if frame is None:
            return

        try:
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                self.get_logger().info('Thoat node')
                self._stop_event.set()
                rclpy.shutdown()

            elif key == ord('r'):
                self.change_rotate_mode()

            elif key == ord('c'):
                self.get_logger().info('Clear target')
                self.clear_target()

        except Exception as e:
            self.get_logger().error(f'Loi display: {e}')

    # ============================================================
    # CLEANUP
    # ============================================================

    def destroy_node(self):
        self._stop_event.set()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloAstraLockNode()
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