import os
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "calib_images")
os.makedirs(SAVE_DIR, exist_ok=True)

# 11x8 칸 체커보드 -> 내부 코너 (10, 7)
CHECKERBOARD = (10, 7)
TARGET_SAMPLES = 40

# 물리적으로 똑바로 달았다면 flip-method 없음 (거꾸로면 nvvidconv flip-method=2 로 변경)
GST_PIPE = (
    "nvv4l2camerasrc device=/dev/video0 ! "
    "video/x-raw(memory:NVMM), format=UYVY, width=2592, height=1800, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 sync=false"
)

# 자동 촬영 파라미터
DETECT_SCALE = 0.35          # 실시간 검출용 축소 비율 (2592x1800 -> 약 907x630)
COOLDOWN_SEC = 1.0           # 연속 촬영 최소 간격 (초)
STILL_THRESH_PX = 10.0       # 흔들림 방지: 직전 프레임 대비 코너 이동량 임계값 (축소 좌표 기준)
MIN_NOVELTY = 0.16           # 기존 샘플과 [x, y, size, skew] 차이가 이 이상일 때 자동 촬영


def compute_board_params(corners, w, h):
    """체커보드 코너로부터 ROS camera_calibration 스타일의 (x, y, size, skew) 계산."""
    pts = corners.reshape(-1, 2)
    cols, rows = CHECKERBOARD

    # 4개 바깥쪽 코너
    p_tl = pts[0]
    p_tr = pts[cols - 1]
    p_bl = pts[(rows - 1) * cols]
    p_br = pts[rows * cols - 1]

    # 중심 위치 (0~1 정규화)
    cx = float(np.mean(pts[:, 0])) / w
    cy = float(np.mean(pts[:, 1])) / h

    # 크기: 보드 면적 / 이미지 면적의 제곱근 (대략 0.1 ~ 0.7)
    quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
    area = cv2.contourArea(quad)
    size = float(np.sqrt(max(area, 1.0) / (w * h)))

    # 기울기(Skew): 대변 길이 비대칭 + 대각선 각도 왜곡
    top = np.linalg.norm(p_tr - p_tl) + 1e-6
    bot = np.linalg.norm(p_br - p_bl) + 1e-6
    left = np.linalg.norm(p_bl - p_tl) + 1e-6
    right = np.linalg.norm(p_br - p_tr) + 1e-6
    v1 = p_tr - p_tl
    v2 = p_bl - p_tl
    cos_angle = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    persp = abs(top - bot) / max(top, bot) + abs(left - right) / max(left, right)
    skew = float(min(1.0, cos_angle * 1.5 + persp * 1.2))

    return np.array([cx, cy, size, skew], dtype=np.float32)


def compute_progress(samples):
    """저장된 샘플들의 X, Y, Size, Skew 커버리지 진행률(0~1) 계산."""
    if len(samples) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(samples, dtype=np.float32)
    # 목표 범위: X(0.15~0.85 -> 0.70), Y(0.15~0.85 -> 0.70), Size(0.18~0.55 -> 0.37), Skew(0.0~0.55 -> 0.55)
    target_spans = [0.65, 0.65, 0.32, 0.45]
    progs = []
    for i, span in enumerate(target_spans):
        actual_span = float(np.max(arr[:, i]) - np.min(arr[:, i]))
        progs.append(min(1.0, actual_span / span))
    return progs


def draw_sidebar(height, progress, current_params, saved_count, auto_mode, status_text, samples):
    """오른쪽에 ROS camera_calibration 스타일 진행률 바 패널 그리기."""
    panel_w = 340
    panel = np.full((height, panel_w, 3), 30, dtype=np.uint8)

    cv2.putText(panel, "CALIBRATION GUIDE", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    mode_color = (80, 230, 80) if auto_mode else (80, 180, 255)
    cv2.putText(
        panel,
        f"AUTO CAPTURE: {'ON' if auto_mode else 'OFF'}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        mode_color,
        2,
    )
    cv2.putText(
        panel,
        f"Saved: {saved_count} / {TARGET_SAMPLES}",
        (20, 102),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
    )

    labels = ["X (Left - Right)", "Y (Up - Down)", "Size (Near - Far)", "Skew (Tilt Angle)"]
    y0 = 145
    bar_x, bar_w, bar_h = 20, 300, 24

    for i, label in enumerate(labels):
        y = y0 + i * 74
        p = progress[i]
        pct = int(p * 100)
        cv2.putText(panel, f"{label}  {pct:3d}%", (bar_x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        # 배경 바
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (70, 70, 70), -1)
        # 진행률 바
        fill_w = int(bar_w * p)
        bar_color = (60, 220, 80) if p >= 0.85 else ((40, 190, 255) if p >= 0.45 else (50, 110, 240))
        if fill_w > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + fill_w, y + bar_h), bar_color, -1)
        # 현재 보드 위치 마커
        if current_params is not None:
            mx = bar_x + int(np.clip(current_params[i], 0.0, 1.0) * bar_w)
            cv2.line(panel, (mx, y - 3), (mx, y + bar_h + 3), (255, 255, 255), 2)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (160, 160, 160), 1)

    # 미니 커버리지 맵 (촬영된 보드 중심 분포)
    map_x, map_y, map_w, map_h = 20, 450, 300, 180
    cv2.putText(panel, "Captured Positions (X-Y Map)", (map_x, map_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h), (50, 50, 50), -1)
    cv2.rectangle(panel, (map_x, map_y), (map_x + map_w, map_y + map_h), (130, 130, 130), 1)
    # 3x3 가이드 선
    for k in (1, 2):
        cv2.line(panel, (map_x + map_w * k // 3, map_y), (map_x + map_w * k // 3, map_y + map_h), (80, 80, 80), 1)
        cv2.line(panel, (map_x, map_y + map_h * k // 3), (map_x + map_w, map_y + map_h * k // 3), (80, 80, 80), 1)
    for s in samples:
        px = map_x + int(np.clip(s[0], 0.0, 1.0) * map_w)
        py = map_y + int(np.clip(s[1], 0.0, 1.0) * map_h)
        cv2.circle(panel, (px, py), 4, (60, 230, 90), -1)
    if current_params is not None:
        px = map_x + int(np.clip(current_params[0], 0.0, 1.0) * map_w)
        py = map_y + int(np.clip(current_params[1], 0.0, 1.0) * map_h)
        cv2.circle(panel, (px, py), 6, (0, 220, 255), 2)

    # 상태 및 단축키 안내
    cv2.putText(panel, status_text, (20, 668), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 255), 2)
    cv2.putText(panel, "[a] Auto ON/OFF  [SPACE] Manual", (20, height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    cv2.putText(panel, "[q] Quit & Finish Capture", (20, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
    return panel


def main():
    cap = cv2.VideoCapture(GST_PIPE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 다른 gst-launch가 켜져 있는지 확인하세요.")

    idx = len([f for f in os.listdir(SAVE_DIR) if f.endswith(".png")])
    samples = []
    auto_mode = True
    last_capture_t = 0.0
    flash_until = 0.0
    prev_corners = None

    print(f"저장 경로: {SAVE_DIR}")
    print("체커보드를 화면에 비추고 상하좌우·원근·기울기를 바꿔주면 자동으로 촬영됩니다.")
    print("[a]: 자동촬영 토글 | [SPACE]: 수동 저장 | [q]: 종료")

    prev_w, prev_h = 1080, 750

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 빠른 실시간 검출을 위해 축소 이미지에서 코너 탐색
        small = cv2.resize(frame, (0, 0), fx=DETECT_SCALE, fy=DETECT_SCALE)
        gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        det_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_FAST_CHECK
        )
        found, corners = cv2.findChessboardCorners(gray_small, CHECKERBOARD, det_flags)

        preview = cv2.resize(frame, (prev_w, prev_h))
        current_params = None
        status_text = "Show checkerboard..."

        now = time.time()
        should_capture = False

        if found:
            # 미리보기 좌표로 변환해 코너 그리기
            sx = prev_w / small.shape[1]
            sy = prev_h / small.shape[0]
            corners_prev = corners.copy()
            corners_prev[:, 0, 0] *= sx
            corners_prev[:, 0, 1] *= sy
            cv2.drawChessboardCorners(preview, CHECKERBOARD, corners_prev, found)

            current_params = compute_board_params(corners, small.shape[1], small.shape[0])

            # 흔들림(모션) 체크
            is_still = False
            if prev_corners is not None:
                motion = float(np.mean(np.linalg.norm(corners - prev_corners, axis=2)))
                is_still = motion < STILL_THRESH_PX
            prev_corners = corners.copy()

            # 기존 샘플과의 차이(새로움) 계산
            if len(samples) == 0:
                novelty = 999.0
            else:
                dists = [float(np.sum(np.abs(current_params - s))) for s in samples]
                novelty = min(dists)

            if auto_mode:
                if now - last_capture_t < COOLDOWN_SEC:
                    status_text = "Captured! Move board..."
                elif not is_still:
                    status_text = "Hold steady..."
                elif novelty < MIN_NOVELTY:
                    status_text = "Move / tilt to new pose"
                else:
                    should_capture = True
            else:
                status_text = "Board OK (Press SPACE)"
        else:
            prev_corners = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            should_capture = True
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"자동 촬영: {'ON' if auto_mode else 'OFF'}")
        elif key == ord("q"):
            break

        if should_capture:
            path = os.path.join(SAVE_DIR, f"img_{idx:03d}.png")
            cv2.imwrite(path, frame)
            print(f"[자동/수동 저장] {path}")
            idx += 1
            if current_params is not None:
                samples.append(current_params)
            last_capture_t = now
            flash_until = now + 0.25
            status_text = f"CAPTURED #{idx}!"

        if now < flash_until:
            cv2.rectangle(preview, (0, 0), (prev_w - 1, prev_h - 1), (0, 255, 0), 10)
            cv2.putText(preview, f"CAPTURED #{idx}", (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        progress = compute_progress(samples)
        if all(p >= 0.85 for p in progress) and idx >= 30:
            status_text = "READY! Press [q] to calibrate"

        sidebar = draw_sidebar(prev_h, progress, current_params, idx, auto_mode, status_text, samples)
        combined = np.hstack([preview, sidebar])
        cv2.imshow("Fisheye Calibration Capture (Auto)", combined)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n촬영 종료 (총 {idx}장). 다음 명령으로 내부 캘리브레이션을 실행하세요:")
    print("  python3 ~/calib/calibrate_fisheye.py")


if __name__ == "__main__":
    main()
