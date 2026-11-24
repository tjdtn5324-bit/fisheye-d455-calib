import os
import time
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "extrinsic_images")
os.makedirs(SAVE_DIR, exist_ok=True)

CHECKERBOARD = (10, 7)
TARGET_PAIRS = 20

# 1) 어안 카메라 (/dev/video0) — capture_calib.py 와 동일한 파이프라인 유지 필수!
FISHEYE_GST = (
    "nvv4l2camerasrc device=/dev/video0 ! "
    "video/x-raw(memory:NVMM), format=UYVY, width=2592, height=1800, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 sync=false"
)

# 2) Intel RealSense D455f RGB (/dev/video6) — 최대 화각(16:10, 90°x65°)
D455_DEV = "/dev/video6"
D455_W, D455_H, D455_FPS = 1280, 800, 30

# 저오차(저 RMS) 정밀 촬영 설정: 두 카메라 모두 0.6초 이상 완전히 정지해야 자동 촬영!
COOLDOWN_SEC = 1.5
STILL_THRESH_PX = 1.8   # 축소 영상 기준 프레임 간 코너 이동량 1.8px 미만일 때만 정지 판정
HOLD_STILL_SEC = 0.6    # 0.6초 연속 정지 유지 시 촬영 (두 카메라 간 지연 오차 제거)
MIN_NOVELTY = 0.16


def compute_board_params(corners, w, h):
    pts = corners.reshape(-1, 2)
    cols, rows = CHECKERBOARD
    p_tl, p_tr = pts[0], pts[cols - 1]
    p_bl, p_br = pts[(rows - 1) * cols], pts[rows * cols - 1]

    cx = float(np.mean(pts[:, 0])) / w
    cy = float(np.mean(pts[:, 1])) / h
    quad = np.array([p_tl, p_tr, p_br, p_bl], dtype=np.float32)
    size = float(np.sqrt(max(cv2.contourArea(quad), 1.0) / (w * h)))

    top = np.linalg.norm(p_tr - p_tl) + 1e-6
    bot = np.linalg.norm(p_br - p_bl) + 1e-6
    left = np.linalg.norm(p_bl - p_tl) + 1e-6
    right = np.linalg.norm(p_br - p_tr) + 1e-6
    v1, v2 = p_tr - p_tl, p_bl - p_tl
    cos_angle = abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6))
    persp = abs(top - bot) / max(top, bot) + abs(left - right) / max(left, right)
    skew = float(min(1.0, cos_angle * 1.5 + persp * 1.2))
    return np.array([cx, cy, size, skew], dtype=np.float32)


def compute_progress(samples):
    if len(samples) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    arr = np.array(samples, dtype=np.float32)
    target_spans = [0.55, 0.55, 0.25, 0.35]
    return [min(1.0, float(np.max(arr[:, i]) - np.min(arr[:, i])) / s) for i, s in enumerate(target_spans)]


def draw_sidebar(height, progress, current_params, saved_count, auto_mode, status_text, still_ratio):
    panel_w = 300
    panel = np.full((height, panel_w, 3), 30, dtype=np.uint8)
    cv2.putText(panel, "EXTRINSIC (STILL)", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
    mode_color = (80, 230, 80) if auto_mode else (80, 180, 255)
    cv2.putText(panel, f"AUTO: {'ON' if auto_mode else 'OFF'}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, mode_color, 2)
    cv2.putText(panel, f"Pairs: {saved_count} / {TARGET_PAIRS}", (18, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

    labels = ["X (L - R)", "Y (U - D)", "Size", "Skew"]
    bar_x, bar_w, bar_h = 18, 264, 20
    for i, label in enumerate(labels):
        y = 122 + i * 58
        p = progress[i]
        cv2.putText(panel, f"{label}  {int(p * 100):3d}%", (bar_x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (70, 70, 70), -1)
        bar_color = (60, 220, 80) if p >= 0.85 else ((40, 190, 255) if p >= 0.45 else (50, 110, 240))
        if int(bar_w * p) > 0:
            cv2.rectangle(panel, (bar_x, y), (bar_x + int(bar_w * p), y + bar_h), bar_color, -1)
        if current_params is not None:
            mx = bar_x + int(np.clip(current_params[i], 0.0, 1.0) * bar_w)
            cv2.line(panel, (mx, y - 3), (mx, y + bar_h + 3), (255, 255, 255), 2)
        cv2.rectangle(panel, (bar_x, y), (bar_x + bar_w, y + bar_h), (160, 160, 160), 1)

    # 정지 유지 게이지 (0.6초 동안 움직이지 않으면 자동 찰칵)
    sy = 368
    cv2.putText(panel, f"Hold Still: {int(still_ratio * 100):3d}%", (bar_x, sy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 255, 255), 1)
    cv2.rectangle(panel, (bar_x, sy), (bar_x + bar_w, sy + 18), (70, 70, 70), -1)
    if int(bar_w * still_ratio) > 0:
        cv2.rectangle(panel, (bar_x, sy), (bar_x + int(bar_w * still_ratio), sy + 18), (0, 255, 200), -1)
    cv2.rectangle(panel, (bar_x, sy), (bar_x + bar_w, sy + 18), (160, 160, 160), 1)

    cv2.putText(panel, status_text, (18, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 255, 255), 2)
    cv2.putText(panel, "[a] Auto  [SPACE] Manual", (18, height - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
    cv2.putText(panel, "[q] Quit", (18, height - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1)
    return panel


def main():
    cap_fish = cv2.VideoCapture(FISHEYE_GST, cv2.CAP_GSTREAMER)
    if not cap_fish.isOpened():
        raise RuntimeError("어안 카메라(/dev/video0)를 열 수 없습니다.")

    cap_d455 = cv2.VideoCapture(D455_DEV, cv2.CAP_V4L2)
    cap_d455.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap_d455.set(cv2.CAP_PROP_FRAME_WIDTH, D455_W)
    cap_d455.set(cv2.CAP_PROP_FRAME_HEIGHT, D455_H)
    cap_d455.set(cv2.CAP_PROP_FPS, D455_FPS)
    if not cap_d455.isOpened():
        cap_fish.release()
        raise RuntimeError(f"D455f RGB({D455_DEV})를 열 수 없습니다.")

    existing = [f for f in os.listdir(SAVE_DIR) if f.startswith("fisheye_") and f.endswith(".png")]
    idx = len(existing)
    samples = []
    auto_mode = True
    last_capture_t = 0.0
    flash_until = 0.0
    prev_corners_f = None
    prev_corners_d = None
    still_since = None

    print(f"저장 경로: {SAVE_DIR}")
    print("체커보드를 새 위치로 옮긴 뒤 0.6초 동안 가만히 멈추면 자동으로 정밀 촬영됩니다!")
    print("[a]: 자동촬영 토글 | [SPACE]: 수동 저장 | [q]: 종료")

    prev_h = 480
    det_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK

    while True:
        ret1, frame_fish = cap_fish.read()
        ret2, frame_d455 = cap_d455.read()
        if not ret1 or not ret2:
            continue

        small_f = cv2.resize(frame_fish, (648, 450))
        small_d = cv2.resize(frame_d455, (640, 400))
        ok_f, corners_f = cv2.findChessboardCorners(cv2.cvtColor(small_f, cv2.COLOR_BGR2GRAY), CHECKERBOARD, det_flags)
        ok_d, corners_d = cv2.findChessboardCorners(cv2.cvtColor(small_d, cv2.COLOR_BGR2GRAY), CHECKERBOARD, det_flags)

        fw = int(frame_fish.shape[1] * prev_h / frame_fish.shape[0])
        dw = int(frame_d455.shape[1] * prev_h / frame_d455.shape[0])
        fish_prev = cv2.resize(frame_fish, (fw, prev_h))
        d455_prev = cv2.resize(frame_d455, (dw, prev_h))

        if ok_f:
            cf = corners_f.copy()
            cf[:, 0, 0] *= fw / small_f.shape[1]
            cf[:, 0, 1] *= prev_h / small_f.shape[0]
            cv2.drawChessboardCorners(fish_prev, CHECKERBOARD, cf, ok_f)
        if ok_d:
            cd = corners_d.copy()
            cd[:, 0, 0] *= dw / small_d.shape[1]
            cd[:, 0, 1] *= prev_h / small_d.shape[0]
            cv2.drawChessboardCorners(d455_prev, CHECKERBOARD, cd, ok_d)

        cv2.putText(
            fish_prev,
            f"Fisheye: {'OK' if ok_f else 'NO BOARD'}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if ok_f else (0, 80, 255),
            2,
        )
        cv2.putText(
            d455_prev,
            f"D455f RGB: {'OK' if ok_d else 'NO BOARD'}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if ok_d else (0, 80, 255),
            2,
        )

        now = time.time()
        current_params = None
        should_capture = False
        still_ratio = 0.0
        status_text = "Show board to BOTH cams"

        if ok_f and ok_d:
            current_params = compute_board_params(corners_d, small_d.shape[1], small_d.shape[0])

            # 두 카메라 모두에서 코너가 움직이지 않았는지 엄격하게 체크
            both_still = False
            if prev_corners_f is not None and prev_corners_d is not None:
                mot_f = float(np.mean(np.linalg.norm(corners_f - prev_corners_f, axis=2)))
                mot_d = float(np.mean(np.linalg.norm(corners_d - prev_corners_d, axis=2)))
                both_still = (mot_f < STILL_THRESH_PX) and (mot_d < STILL_THRESH_PX)
            prev_corners_f = corners_f.copy()
            prev_corners_d = corners_d.copy()

            novelty = 999.0 if len(samples) == 0 else min(float(np.sum(np.abs(current_params - s))) for s in samples)

            if auto_mode:
                if now - last_capture_t < COOLDOWN_SEC:
                    still_since = None
                    status_text = "Captured! Move board..."
                elif novelty < MIN_NOVELTY:
                    still_since = None
                    status_text = "Move / tilt to new pose"
                elif not both_still:
                    still_since = None
                    status_text = "Hold completely still..."
                else:
                    if still_since is None:
                        still_since = now
                    elapsed = now - still_since
                    still_ratio = min(1.0, elapsed / HOLD_STILL_SEC)
                    if elapsed >= HOLD_STILL_SEC:
                        should_capture = True
                        still_since = None
                    else:
                        status_text = "Hold still... almost!"
            else:
                still_since = None
                status_text = "BOTH OK (Press SPACE)"
        else:
            prev_corners_f = None
            prev_corners_d = None
            still_since = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            should_capture = True
        elif key == ord("a"):
            auto_mode = not auto_mode
        elif key == ord("q"):
            break

        if should_capture:
            p_fish = os.path.join(SAVE_DIR, f"fisheye_{idx:03d}.png")
            p_d455 = os.path.join(SAVE_DIR, f"d455_{idx:03d}.png")
            cv2.imwrite(p_fish, frame_fish)
            cv2.imwrite(p_d455, frame_d455)
            print(f"[정지 페어 저장 {idx:03d}] {os.path.basename(p_fish)} & {os.path.basename(p_d455)}")
            idx += 1
            if current_params is not None:
                samples.append(current_params)
            last_capture_t = now
            flash_until = now + 0.3
            status_text = f"CAPTURED PAIR #{idx}!"

        views = np.hstack([fish_prev, d455_prev])
        if now < flash_until:
            cv2.rectangle(views, (0, 0), (views.shape[1] - 1, views.shape[0] - 1), (0, 255, 0), 8)

        progress = compute_progress(samples)
        if all(p >= 0.85 for p in progress) and idx >= 18:
            status_text = "READY! Press [q] to calibrate"

        sidebar = draw_sidebar(prev_h, progress, current_params, idx, auto_mode, status_text, still_ratio)
        cv2.imshow("Extrinsic Stereo Capture (Still Mode)", np.hstack([views, sidebar]))

    cap_fish.release()
    cap_d455.release()
    cv2.destroyAllWindows()
    print(f"\n촬영 종료 (총 {idx}쌍). 다음 명령으로 외부 캘리브레이션을 실행하세요:")
    print("  python3 ~/calib/calibrate_extrinsic.py")


if __name__ == "__main__":
    main()
