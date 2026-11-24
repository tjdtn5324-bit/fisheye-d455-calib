import glob
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "calib_images")

# 11x8 칸 체커보드 -> 내부 코너 (10, 7)
CHECKERBOARD = (10, 7)
SQUARE_W_MM = 266.0 / 11.0  # 24.1818 mm
SQUARE_H_MM = 193.0 / 8.0   # 24.1250 mm

images = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
assert len(images) >= 10, f"이미지가 너무 적습니다: {len(images)}장 ({IMG_DIR})"

# 3D 코너 좌표 (mm 단위, Z=0 평면)
objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
grid = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp[0, :, 0] = grid[:, 0] * SQUARE_W_MM
objp[0, :, 1] = grid[:, 1] * SQUARE_H_MM

objpoints = []
imgpoints = []
valid_files = []
img_shape = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]  # (W, H) = (2592, 1800)

    # 어안 렌즈에서 정확도가 높은 SB 검출기 우선 사용
    ret, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD,
        cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
    )
    if not ret:
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret:
            corners = cv2.cornerSubPix(
                gray,
                corners,
                (11, 11),
                (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),
            )

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners.astype(np.float64))
        valid_files.append(fname)
        print(f"[OK]   {os.path.basename(fname)}")
    else:
        print(f"[FAIL] {os.path.basename(fname)} (코너 미검출 - 제외)")

print(f"\n총 {len(images)}장 중 {len(valid_files)}장 검출 성공. Fisheye 캘리브레이션 계산 중...")

K = np.zeros((3, 3), dtype=np.float64)
D = np.zeros((4, 1), dtype=np.float64)
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

flags = (
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    | cv2.fisheye.CALIB_CHECK_COND
    | cv2.fisheye.CALIB_FIX_SKEW
)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

try:
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, flags, criteria
    )
except cv2.error:
    print("일부 극단적 각도 이미지로 CALIB_CHECK_COND 실패 -> 조건 완화 후 재계산")
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints, imgpoints, img_shape, K, D, rvecs, tvecs, flags, criteria
    )

# 이미지별 재투영 오차 출력 (튀는 사진 확인용)
per_img_err = []
for i in range(len(objpoints)):
    proj, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    err = cv2.norm(imgpoints[i], proj.reshape(-1, 1, 2), cv2.NORM_L2) / np.sqrt(len(proj[0]))
    per_img_err.append((err, valid_files[i]))
per_img_err.sort(reverse=True)

print("\n===== Fisheye Calibration 결과 =====")
print(f"이미지 크기 : {img_shape}")
print(f"RMS 오차    : {rms:.4f} px  (0.5 이하 권장)")
print(f"K (원본):\n{K}")
print(f"D (k1..k4): {D.ravel()}")
print("\n오차 큰 상위 5장 (1.0px 넘으면 삭제 후 재계산 권장):")
for e, f in per_img_err[:5]:
    print(f"  {e:.3f} px - {os.path.basename(f)}")

# TRGv2용 Pinhole K_new 생성 (balance=0.0: 검은 테두리 제거 / 0.3~0.5: 화각 더 보존)
BALANCE = 0.0
K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, img_shape, np.eye(3), balance=BALANCE
)
print(f"\nK_new (TRGv2에 넣을 Pinhole 내부 행렬, balance={BALANCE}):\n{K_new}")

# 파라미터 저장
out_npz = os.path.join(BASE_DIR, "calib_fisheye.npz")
np.savez(
    out_npz,
    K=K,
    D=D,
    K_new=K_new,
    img_shape=np.array(img_shape),
    rms=rms,
    balance=BALANCE,
)
print(f"\n저장 완료: {out_npz}")

# 첫 번째 사진으로 왜곡 보정 전/후 비교 이미지 저장
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), K_new, img_shape, cv2.CV_16SC2
)
sample = cv2.imread(valid_files[0])
undist = cv2.remap(
    sample, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
)
compare = np.hstack([cv2.resize(sample, (960, 666)), cv2.resize(undist, (960, 666))])
out_jpg = os.path.join(BASE_DIR, "undistort_compare.jpg")
cv2.imwrite(out_jpg, compare)
print(f"비교 이미지 저장: {out_jpg} (왼쪽=원본, 오른쪽=왜곡보정)")
