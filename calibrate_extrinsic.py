import glob
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "extrinsic_images")
FISH_NPZ = os.path.join(BASE_DIR, "calib_fisheye.npz")
D455_NPZ = os.path.join(BASE_DIR, "calib_d455.npz")
OUT_NPZ = os.path.join(BASE_DIR, "calib_extrinsic.npz")

# calibrate_fisheye.py 와 동일한 체커보드 설정
CHECKERBOARD = (10, 7)
SQUARE_W_MM = 266.0 / 11.0  # 24.1818 mm
SQUARE_H_MM = 193.0 / 8.0   # 24.1250 mm
OUTLIER_THRESH_PX = 3.0


def detect_corners(gray):
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if ret:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4),
        )
    else:
        ret, corners = cv2.findChessboardCornersSB(
            gray, CHECKERBOARD, cv2.CALIB_CB_ACCURACY
        )
    if not ret:
        return False, None

    # 두 카메라 간 180도 코너 순서 뒤집힘 방지: 항상 좌->우(+x), 상->하(+y) 정렬
    cols, rows = CHECKERBOARD
    pts = corners.reshape(rows, cols, 2)
    if pts[0, -1, 0] < pts[0, 0, 0]:
        pts = pts[:, ::-1, :]
    if pts[-1, 0, 1] < pts[0, 0, 1]:
        pts = pts[::-1, :, :]
    corners = np.ascontiguousarray(pts.reshape(-1, 1, 2), dtype=np.float32)
    return True, corners


def rotation_matrix_to_euler_deg(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.degrees([x, y, z])


def compute_pair_errors(objpoints, pts_f, pts_d, K_new, K_d455, D_d455, R, T):
    """각 페어에서 Fisheye PnP -> R, T 변환 -> D455f 재투영 오차(px) 계산."""
    errs = []
    for obj, pf, pd in zip(objpoints, pts_f, pts_d):
        ok, rvec_f, tvec_f = cv2.solvePnP(obj, pf, K_new, np.zeros((5, 1)))
        if not ok:
            errs.append(999.0)
            continue
        R_f, _ = cv2.Rodrigues(rvec_f)
        R_d = R @ R_f
        t_d = R @ tvec_f + T
        rvec_d, _ = cv2.Rodrigues(R_d)
        proj_d, _ = cv2.projectPoints(obj, rvec_d, t_d, K_d455, D_d455)
        e = cv2.norm(pd, proj_d, cv2.NORM_L2) / np.sqrt(len(obj))
        errs.append(float(e))
    return errs


def main():
    assert os.path.exists(FISH_NPZ), f"먼저 1단계(calibrate_fisheye.py)를 완료하세요: {FISH_NPZ} 없음"
    fish_data = np.load(FISH_NPZ)
    K_fish = fish_data["K"].astype(np.float64)
    D_fish = fish_data["D"].astype(np.float64)
    K_new = fish_data["K_new"].astype(np.float64)

    fish_files = sorted(glob.glob(os.path.join(IMG_DIR, "fisheye_*.png")))
    assert len(fish_files) >= 10, f"페어 이미지가 너무 적습니다: {len(fish_files)}쌍 ({IMG_DIR})"

    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    grid = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp[:, 0] = grid[:, 0] * SQUARE_W_MM
    objp[:, 1] = grid[:, 1] * SQUARE_H_MM

    objpoints = []
    imgpoints_fish_undist = []
    imgpoints_d455 = []
    valid_names = []
    fish_shape = None
    d455_shape = None

    for f_path in fish_files:
        idx_str = os.path.basename(f_path).replace("fisheye_", "")
        d_path = os.path.join(IMG_DIR, f"d455_{idx_str}")
        if not os.path.exists(d_path):
            print(f"[SKIP] 짝 파일 없음: {d_path}")
            continue

        gray_f = cv2.cvtColor(cv2.imread(f_path), cv2.COLOR_BGR2GRAY)
        gray_d = cv2.cvtColor(cv2.imread(d_path), cv2.COLOR_BGR2GRAY)
        fish_shape = gray_f.shape[::-1]
        d455_shape = gray_d.shape[::-1]

        ret_f, corners_f = detect_corners(gray_f)
        ret_d, corners_d = detect_corners(gray_d)

        if ret_f and ret_d:
            corners_f_undist = cv2.fisheye.undistortPoints(
                corners_f.astype(np.float64).reshape(-1, 1, 2),
                K_fish,
                D_fish,
                R=np.eye(3),
                P=K_new,
            ).astype(np.float32)

            objpoints.append(objp)
            imgpoints_fish_undist.append(corners_f_undist)
            imgpoints_d455.append(corners_d.astype(np.float32))
            valid_names.append(idx_str)
            print(f"[OK]   pair {idx_str}")
        else:
            print(f"[FAIL] pair {idx_str} (fisheye={ret_f}, d455={ret_d})")

    print(f"\n총 {len(fish_files)}쌍 중 {len(valid_names)}쌍 검출 성공.")
    assert len(valid_names) >= 10, "유효한 페어가 10쌍 미만입니다."

    # D455f RGB 내부 파라미터 로드 또는 계산
    if os.path.exists(D455_NPZ):
        d455_data = np.load(D455_NPZ)
        K_d455 = d455_data["K"].astype(np.float64)
        D_d455 = d455_data["D"].astype(np.float64)
        print(f"기존 D455f 내부 파라미터 로드: {D455_NPZ}")
    else:
        print("D455f RGB 내부 캘리브레이션 계산 중 (Pinhole 모델)...")
        rms_d455, K_d455, D_d455, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_d455, d455_shape, None, None
        )
        print(f"D455f RGB 내부 RMS 오차: {rms_d455:.4f} px")
        np.savez(D455_NPZ, K=K_d455, D=D_d455, img_shape=np.array(d455_shape), rms=rms_d455)

    # 1차 스테레오 캘리브레이션
    dist_fish_undist = np.zeros((5, 1), dtype=np.float64)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    rms_all, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_fish_undist,
        imgpoints_d455,
        K_new,
        dist_fish_undist,
        K_d455,
        D_d455,
        d455_shape,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    pair_errs = compute_pair_errors(objpoints, imgpoints_fish_undist, imgpoints_d455, K_new, K_d455, D_d455, R, T)
    ranked = sorted(zip(pair_errs, range(len(valid_names)), valid_names), reverse=True)
    print(f"\n1차 Stereo RMS (전체 {len(valid_names)}쌍): {rms_all:.4f} px")
    print("오차 큰 상위 10쌍:")
    for e, _, name in ranked[:10]:
        print(f"  {e:6.2f} px - pair {name}")

    # 이상치(움직임/타이밍 어긋남 등) 자동 제거 후 2차 재계산
    inliers = [i for e, i, _ in ranked if e <= OUTLIER_THRESH_PX]
    if len(inliers) >= 10 and len(inliers) < len(valid_names):
        print(f"\n오차 <= {OUTLIER_THRESH_PX} px 인라이어 {len(inliers)}쌍으로 2차 정밀 재계산...")
        rms_stereo, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            [objpoints[i] for i in inliers],
            [imgpoints_fish_undist[i] for i in inliers],
            [imgpoints_d455[i] for i in inliers],
            K_new,
            dist_fish_undist,
            K_d455,
            D_d455,
            d455_shape,
            criteria=criteria,
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        used_count = len(inliers)
    else:
        rms_stereo = rms_all
        used_count = len(valid_names)

    T_d455_from_fish = np.eye(4, dtype=np.float64)
    T_d455_from_fish[:3, :3] = R
    T_d455_from_fish[:3, 3] = T.ravel()
    T_fish_from_d455 = np.linalg.inv(T_d455_from_fish)
    baseline_mm = float(np.linalg.norm(T))
    euler_deg = rotation_matrix_to_euler_deg(R)

    print(f"\n===== 최종 Extrinsic Calibration 결과 (Fisheye -> D455f RGB, {used_count}쌍 사용) =====")
    print(f"Stereo RMS 오차 : {rms_stereo:.4f} px")
    print(f"R (3x3):\n{R}")
    print(f"Euler (rx, ry, rz) [deg]: {euler_deg}")
    print(f"T [mm] (tx, ty, tz): {T.ravel()}")
    print(f"T [m]  (tx, ty, tz): {(T.ravel() / 1000.0)}")
    print(f"두 카메라 간 거리 (Baseline): {baseline_mm:.2f} mm ({baseline_mm / 10.0:.2f} cm)")
    print(f"\n4x4 변환행렬 T_d455_from_fish (mm):\n{T_d455_from_fish}")

    np.savez(
        OUT_NPZ,
        R=R,
        T_mm=T,
        T_m=T / 1000.0,
        T_d455_from_fish_mm=T_d455_from_fish,
        T_fish_from_d455_mm=T_fish_from_d455,
        E=E,
        F=F,
        rms_stereo=rms_stereo,
        K_fish=K_fish,
        D_fish=D_fish,
        K_fish_new=K_new,
        K_d455=K_d455,
        D_d455=D_d455,
        fish_shape=np.array(fish_shape),
        d455_shape=np.array(d455_shape),
    )
    print(f"\n저장 완료: {OUT_NPZ}")


if __name__ == "__main__":
    main()
