"""
第4课：相机 + 激光雷达融合（后融合 / 目标级融合）

流程：
  1. 坐标系：雷达系(=机器人系) → 相机系 → 像素。需要两组参数：
       内参 K：焦距 fx, fy 和主点 cx, cy（相机自己的属性，标定板标出来）
       外参 R, t：相机相对雷达的旋转和平移（两传感器联合标定）
  2. 视觉模型给出 2D 框（“是什么”）
  3. 把点云投到图像上，取落在框里的点，算出距离（“在哪、多远”）
  4. 实验：外参差几度会怎样？时间不同步会怎样？

运行：python3 lesson4_fusion.py   （与第1、2课文件同目录）
"""
import numpy as np
import cv2
import lesson1_pointcloud_basics as L
import lesson2_zone_ground as Z

rng = np.random.default_rng(4)
IMG_W, IMG_H = 640, 480
K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])   # 内参

# 机器人前方的障碍物：(名字, 中心xy, 尺寸, 颜色BGR)
OBJECTS = [("rock", (3.0, 0.8), (0.4, 0.3, 0.25), (140, 140, 140)),
           ("toy", (2.0, -0.6), (0.3, 0.2, 0.12), (40, 40, 220)),
           ("box_far", (5.0, -2.0), (0.5, 0.4, 0.35), (40, 180, 230))]


# ---------------------------------------------------------------
# 场景：沿用第2课的起伏地形，障碍物放在机器人前方
# ---------------------------------------------------------------
def make_scene():
    Z.rng = np.random.default_rng(1)
    n = 50000
    r = Z.rng.uniform(0.3, 7.0, n)
    th = Z.rng.uniform(-np.pi, np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    pts = [np.column_stack([x, y, Z.terrain(x, y) + Z.rng.normal(0, 0.015, n)])]
    lab = [np.full(n, -1)]
    for i, (_, c, s, _) in enumerate(OBJECTS):
        b = Z.box(c, np.array(s), 1200)
        pts.append(b); lab.append(np.full(len(b), i))
    return np.vstack(pts), np.concatenate(lab)


# ---------------------------------------------------------------
# 外参：相机装在机器人前方 0.2m、离地 0.35m，朝 +x 看，向下俯 20°
# 相机坐标系约定（OpenCV）：z 向前，x 向右，y 向下
# ---------------------------------------------------------------
def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def camera_pose(pitch_deg=20.0, yaw_err=0.0, pitch_err=0.0):
    p = np.radians(pitch_deg + pitch_err)
    z_cam = np.array([np.cos(p), 0, -np.sin(p)])        # 光轴方向（机器人系中）
    x_cam = np.array([0, -1.0, 0])                       # 图像右方 = 机器人右侧(-y)
    y_cam = np.cross(z_cam, x_cam)                       # 图像下方
    R_base_cam = rot_z(yaw_err) @ np.column_stack([x_cam, y_cam, z_cam])
    t_base_cam = np.array([0.2, 0.0, 0.35])
    # 机器人系点 → 相机系点：p_cam = R_cam_base @ p + t_cam_base
    R_cam_base = R_base_cam.T
    t_cam_base = -R_cam_base @ t_base_cam
    return R_cam_base, t_cam_base


def project(pts, R, t):
    """手写投影：机器人系 3D 点 → 像素 (u, v) 和深度 z"""
    pc = pts @ R.T + t
    z = pc[:, 2]
    uv = (pc @ K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
    valid = (z > 0.2) & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
    return uv, z, valid


# ---------------------------------------------------------------
# 合成“相机图像”：按深度从远到近画点（近处盖住远处）
# ---------------------------------------------------------------
def render(pts, lab, R, t):
    img = np.full((IMG_H, IMG_W, 3), (235, 206, 160), np.uint8)      # 天空
    uv, z, valid = project(pts, R, t)
    # 先画一层地面底色，避免近处稀疏点之间漏出天空
    gm = valid & (lab < 0)
    hull = cv2.convexHull(uv[gm].astype(np.int32))
    cv2.fillConvexPoly(img, hull, (40, 130, 50))
    order = np.argsort(-z)
    for i in order:
        if not valid[i]:
            continue
        if lab[i] >= 0:
            col = OBJECTS[lab[i]][3]
        else:
            g = int(np.clip(110 + 120 * pts[i, 2], 60, 200))
            col = (40, g, 50)
        cv2.circle(img, (int(uv[i, 0]), int(uv[i, 1])), int(np.clip(4 / z[i], 1, 4)), col, -1)
    return img


def fake_detector(pts, lab, R, t, noise_px=3):
    """假装是第3课那样的视觉模型：给出每个障碍物的 2D 框（加一点框的误差）"""
    uv, _, valid = project(pts, R, t)
    boxes = []
    for i, (name, *_ ) in enumerate(OBJECTS):
        m = valid & (lab == i)
        if m.sum() < 5:
            continue
        x0, y0 = uv[m].min(0) - rng.normal(0, noise_px, 2)
        x1, y1 = uv[m].max(0) + rng.normal(0, noise_px, 2)
        boxes.append((name, i, (x0, y0, x1, y1)))
    return boxes


# ---------------------------------------------------------------
# 融合：框里的点云 → 障碍物位置
# ---------------------------------------------------------------
def fuse(box, pts, uv, z, valid, nonground, method):
    x0, y0, x1, y1 = box
    inbox = valid & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    if method == "框内全部点取中位数":
        sel = inbox
    elif method == "去掉地面点再取中位数":
        sel = inbox & nonground
    else:  # "取框内最近的一簇"：按深度排序，从最近处找第一个足够密的深度簇
        idx = np.nonzero(inbox & nonground)[0]
        if len(idx) < 3:
            idx = np.nonzero(inbox)[0]
        if len(idx) == 0:
            return None, inbox
        zs = z[idx]
        zmin = np.percentile(zs, 5)
        sel = np.zeros(len(pts), bool)
        sel[idx[zs < zmin + 0.4]] = True
    if sel.sum() == 0:
        return None, inbox
    return np.median(pts[sel, :2], axis=0), inbox


def run(pts, lab, nonground, boxes, R, t, methods, tag=""):
    uv, z, valid = project(pts, R, t)
    rows = []
    for name, i, box in boxes:
        gt = np.array(OBJECTS[i][1])
        x0, y0, x1, y1 = box
        inbox = valid & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
        purity = (lab[inbox] == i).mean() if inbox.any() else 0
        line = f"  {tag}{name:8s} 真实距离 {np.linalg.norm(gt):.2f}m | 框内点 {inbox.sum():4d}，属于该物体 {purity:5.1%}"
        for mth in methods:
            est, _ = fuse(box, pts, uv, z, valid, nonground, mth)
            err = np.linalg.norm(est - gt) if est is not None else np.nan
            line += f" | {mth}: 误差 {err:.2f}m"
            rows.append((name, mth, err))
        print(line)
    return uv, valid


def main():
    pts, lab = make_scene()
    R, t = camera_pose()

    # 用 OpenCV 的 projectPoints 验证手写投影
    rvec, _ = cv2.Rodrigues(R)
    uv_cv, _ = cv2.projectPoints(pts[:2000], rvec, t, K, None)
    uv_me, _, v = project(pts[:2000], R, t)
    print(f"手写投影 vs cv2.projectPoints 最大差 {np.abs(uv_cv[:, 0][v] - uv_me[v]).max():.2e} 像素\n")

    nonground = ~Z.zone_ground(pts, max_elev=9)[0]       # 第2课的分区地面分割
    img = render(pts, lab, R, t)
    boxes = fake_detector(pts, lab, R, t)

    print("== 1 三种融合方法（外参准确）==")
    methods = ["框内全部点取中位数", "去掉地面点再取中位数", "取框内最近的一簇"]
    uv, valid = run(pts, lab, nonground, boxes, R, t, methods)

    print("\n== 2 外参偏航角有误差（标定不准，或机器人转弯时相机和雷达时间不同步）==")
    best = ["取框内最近的一簇"]
    for err in (1, 3, 6):
        Re, te = camera_pose(yaw_err=err)
        run(pts, lab, nonground, boxes, Re, te, best, tag=f"[偏{err}°] ")
    print("  换算：机器人以 30°/s 转弯，相机和雷达时间差 100ms → 偏差 3°")

    print("\n== 3 外参俯仰角有误差（比如机器人颠簸、相机支架松动）==")
    for err in (1, 3):
        Re, te = camera_pose(pitch_err=err)
        run(pts, lab, nonground, boxes, Re, te, best, tag=f"[俯仰{err}°] ")

    # ---- 画图：左 = 外参准确，右 = 偏 3° ----
    out = []
    for yaw in (0, 3):
        Re, te = camera_pose(yaw_err=yaw)
        vis = img.copy()
        uvp, zp, vp = project(pts, Re, te)
        m = vp & nonground
        for (u, vv), d in zip(uvp[m], zp[m]):
            c = cv2.applyColorMap(np.uint8([[np.clip(255 - d * 40, 0, 255)]]), cv2.COLORMAP_JET)[0, 0]
            cv2.circle(vis, (int(u), int(vv)), 2, tuple(int(x) for x in c), -1)
        for name, i, (x0, y0, x1, y1) in boxes:
            cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 2)
            cv2.putText(vis, name, (int(x0), int(y0) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"extrinsic yaw error {yaw} deg (dots = non-ground lidar, color = depth)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        out.append(vis)
    cv2.imwrite("lesson4_result.png", np.vstack(out))
    print("\n图已保存: lesson4_result.png（上 = 外参准确，下 = 偏 3°）")


if __name__ == "__main__":
    main()
