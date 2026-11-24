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
