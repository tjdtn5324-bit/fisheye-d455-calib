# Fisheye + RealSense D455f Calibration Toolkit

ROS `camera_calibration` 스타일의 실시간 체커보드 인식 + 자동 촬영(X/Y/Size/Skew 진행률 바 & 정지 유지 게이지)을 지원하는 **단안 어안 카메라(`/dev/video0`) ↔ Intel RealSense D455f RGB(`/dev/video6`)** 내부(Intrinsic) · 외부(Extrinsic) 캘리브레이션 도구입니다.

## 주요 기능

- **1단계 내부 캘리브레이션 (`capture_calib.py` → `calibrate_fisheye.py`)**
  - Jetson GStreamer (`nvv4l2camerasrc`, `UYVY 2592×1800 @ 30fps`) 기반 원본 해상도 촬영
  - 실시간 체커보드 검출 + 우측 `X / Y / Size / Skew` 커버리지 바 & X-Y 분포 미니맵
  - OpenCV `cv2.fisheye.calibrate`로 `K`, `D`, `K_new` 계산 및 `calib_fisheye.npz`, `undistort_compare.jpg` 저장
- **2단계 외부 캘리브레이션 (`capture_extrinsic.py` → `calibrate_extrinsic.py`)**
  - `pyrealsense2` 없이 V4L2(`/dev/video6`, `YUYV 1280×800 @ 30fps` 16:10 풀 화각)로 D455f RGB 직접 수신
  - **Hold Still 정지 유지 모드**: 두 카메라 모두 0.6초 이상 정지(`코너 이동 < 1.8 px`)할 때만 자동 페어 촬영 → 비동기 스트림 간 모션 오차 최소화
  - 체커보드 코너 순서 자동 정렬(180° 반전 방지) + 오차 큰 이상치 페어 자동 필터링 후 `cv2.stereoCalibrate` 계산
  - D455f RGB 내부(`calib_d455.npz`) 및 두 카메라 간 `R`, `T`, `T_d455_from_fish_mm`, `T_fish_from_d455_mm` (`calib_extrinsic.npz`) 저장

## 환경 요구사항

- Linux (NVIDIA Jetson 등)
- Python 3
- OpenCV (`GStreamer` 지원 빌드), NumPy

## 사용 방법

### 1. 어안 카메라 내부 캘리브레이션
```bash
# 자동 촬영 (35~50장, 우측 바를 채운 뒤 q로 종료)
python3 capture_calib.py

# 내부 파라미터 계산 (calib_fisheye.npz 생성)
python3 calibrate_fisheye.py
```

### 2. 어안 ↔ D455f RGB 외부 캘리브레이션
> 두 카메라를 실제 사용할 위치에 단단히 고정한 상태에서 진행하세요.

```bash
# 정지 유지 동시 페어 촬영 (20~25쌍, Hold Still 100% 채우며 촬영 후 q)
python3 capture_extrinsic.py

# D455f 내부 + 어안<->D455f 외부 캘리브레이션 계산 (calib_extrinsic.npz 생성)
python3 calibrate_extrinsic.py
```

### 촬영 단축키
- `[a]`: 자동 촬영 ON/OFF 토글
- `[SPACE]`: 수동 즉시 촬영
- `[q]`: 종료

## 결과 파일 불러오기

```python
import numpy as np

# 1) 어안 내부 파라미터
fish = np.load("calib_fisheye.npz")
K_fish, D_fish, K_new = fish["K"], fish["D"], fish["K_new"]

# 2) 외부 변환 행렬 (Fisheye -> D455f RGB)
ext = np.load("calib_extrinsic.npz")
R = ext["R"]                                  # (3, 3)
T_m = ext["T_m"]                              # (3, 1) meters (mm는 ext["T_mm"])
T_d455_from_fish = ext["T_d455_from_fish_mm"] # (4, 4) mm
T_fish_from_d455 = ext["T_fish_from_d455_mm"] # (4, 4) mm (역변환)

```



````markdown
# AVS300 + OX05B1S-RGBIR Camera Setup Guide

Telelian **AVS300 (NVIDIA Jetson AGX Thor)** 보드에서 **OX05B1S-RGBIR (GMSL2 / FAKRA)** 카메라를 연결하고 실행하는 매뉴얼입니다.

---

## 1. Environment & Hardware Specs

| 항목 | 사양 |
| :--- | :--- |
| **Board** | Telelian AVS300 (NVIDIA Jetson AGX Thor, L4T R39.2.1) |
| **Camera Sensor** | OmniVision OX05B1S-RGBIR (Sensing SG5-OX05BC, GMSL2 6G) |
| **Interface** | GMSL2 (FAKRA Coaxial Cable) |
| **Device Tree Overlay** | `AVS300 4x2-sg5-ox05bc` (`/boot/tegra264-avs300-camera-gmsl-4x2-6g-sg5-ox05bc-overlay.dtbo`) |
| **Output Format** | `UYVY 4:2:2`, `2592x1800 @ 30 fps` |
| **Video Nodes** | `/dev/video0` (RGB), `/dev/video1` (IR) |

---

## 2. Hardware Connection

1. FAKRA 동축 케이블을 AVS300의 **CAM0 포트**에 딸깍 소리가 날 때까지 끝까지 체결합니다.
2. **반드시 케이블을 연결한 상태에서 전원을 켜거나 재부팅**해야 합니다. (GMSL2는 핫플러그 미지원)

```bash
sudo reboot
```

---

## 3. Verify Camera Recognition

부팅 후 비디오 장치와 커널 드라이버 인식 상태를 확인합니다.

```bash
# 1. 비디오 장치 확인 (/dev/video0: RGB, /dev/video1: IR)
ls -l /dev/video*

# 2. 지원 포맷 및 해상도 확인
v4l2-ctl -d /dev/video0 --list-formats-ext
v4l2-ctl -d /dev/video1 --list-formats-ext

# 3. 드라이버 및 GMSL2 링크 로그 확인
sudo dmesg | grep -iE "rgbir|max929"
```

---

## 4. Camera Preview (GStreamer)

종료 시 터미널에서 `Ctrl + C`를 누릅니다.  
*(SSH로 접속해 보드에 연결된 모니터에 띄우는 경우 먼저 `export DISPLAY=:0` 실행)*

### RGB (`/dev/video0`) 원본 화질 출력
```bash
gst-launch-1.0 nvv4l2camerasrc device=/dev/video0 ! \
  'video/x-raw(memory:NVMM), format=UYVY, width=2592, height=1800, framerate=30/1' ! \
  nvvidconv ! 'video/x-raw, format=I420' ! \
  xvimagesink sync=false
```

### IR (`/dev/video1`) 원본 화질 출력
```bash
gst-launch-1.0 nvv4l2camerasrc device=/dev/video1 ! \
  'video/x-raw(memory:NVMM), format=UYVY, width=2592, height=1800, framerate=30/1' ! \
  nvvidconv ! 'video/x-raw, format=I420' ! \
  xvimagesink sync=false
```

### 자주 사용하는 옵션
- **화면이 180° 거꾸로 나올 때:** `nvvidconv !` → `nvvidconv flip-method=2 !`
- **창 크기 축소 출력:** `'video/x-raw, format=I420'` → `'video/x-raw, format=I420, width=960, height=666'`

---

## 5. OpenCV (Python) Pipeline Example

```python
import cv2

gst_pipe = (
    "nvv4l2camerasrc device=/dev/video0 ! "
    "video/x-raw(memory:NVMM), format=UYVY, width=2592, height=1800, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! "
    "appsink drop=1 sync=false"
)

cap = cv2.VideoCapture(gst_pipe, cv2.CAP_GSTREAMER)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow("OX05B1S RGB", cv2.resize(frame, (1296, 900)))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 6. Device Tree Overlay Check (참고)

현재 보드에 적용된 오버레이 확인 및 변경 방법:

```bash
# 현재 적용된 오버레이 확인
cat /boot/extlinux/extlinux.conf | grep OVERLAYS

# 선택 가능한 오버레이 목록 확인
sudo /opt/nvidia/jetson-io/config-by-hardware.py -l

# OX05B1S 오버레이 적용 (이미 적용되어 있다면 불필요)
sudo /opt/nvidia/jetson-io/config-by-hardware.py -n "AVS300 4x2-sg5-ox05bc"
sudo reboot
```

---

## 7. Troubleshooting

| 증상 | 원인 | 해결 방법 |
| :--- | :--- | :--- |
| `/dev/video*`가 생성되지 않음 | 부팅 후 케이블 연결 또는 체결 불량 | FAKRA 케이블 재체결 후 `sudo reboot` |
| `Device '/dev/video0' is busy` | 이전 카메라 프로세스가 점유 중 | `pkill -9 gst-launch-1.0` 실행 후 재시도 (`sudo fuser -v /dev/video0`로 점유 프로세스 확인) |
| SSH 실행 시 창이 뜨지 않음 | `DISPLAY` 환경변수 미설정 | `export DISPLAY=:0` 입력 후 재실행 |
| 화면이 상하좌우 반전됨 | 카메라 장착 방향 | GStreamer 파이프라인에 `nvvidconv flip-method=2` 추가 또는 카메라 정방향 재장착 |
| `v4l2-ctl` 명령어 없음 | `v4l-utils` 미설치 | `sudo apt update && sudo apt install -y v4l-utils` |
````
