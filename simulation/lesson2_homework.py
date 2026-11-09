"""
第2课作业讲解
  作业1：块变大会怎样
  作业2：自适应高度检查（用相邻已确认地面做参照，A-GLE 思路的简化版）
  作业3：机器人自己站在斜坡上（雷达整体倾斜），以及用 IMU 校正
运行：python3 lesson2_homework.py   （与第1、2课文件同目录）
"""
import numpy as np
import lesson1_pointcloud_basics as L
import lesson2_zone_ground as Z

# ---------- 准备同一份数据 ----------
pts, gt0 = Z.make_scene()
p2, inv = L.voxel_downsample(pts, 0.05)
g2 = np.array([np.bincount(gt0[inv == v], minlength=5).argmax() for v in range(len(p2))])
k = L.statistical_outlier_removal(p2)
P, gt = p2[k], g2[k]

print("====== 基线 ======")
g, _ = Z.zone_ground(P)
Z.evaluate("第2课原版            ", g, gt, 0)

# ---------- 作业1：最内区域 16 扇区 -> 4 扇区，并把所有区域都切粗 ----------
print("\n====== 作业1：块变大 ======")
big_inner = [(0.3, 2.0, 4, 4), (2.0, 4.0, 4, 32), (4.0, 7.0, 3, 32)]
g, _ = Z.zone_ground(P, zones=big_inner)
Z.evaluate("最内区域 16→4 扇区    ", g, gt, 0)
all_big = [(0.3, 2.0, 1, 4), (2.0, 4.0, 1, 4), (4.0, 7.0, 1, 4)]
g, _ = Z.zone_ground(P, zones=all_big)
Z.evaluate("全部切成大块(12块)    ", g, gt, 0)


# ---------- 作业2：自适应高度检查 ----------
def zone_ground_adaptive(pts, num_lpr=10, th_seed=0.10, th_dist=0.05, iters=3, min_pts=10,
                         min_upright=np.cos(np.radians(40)), max_flat=0.002,
                         max_jump=0.15, ref_radius=1.5, zones=Z.ZONES):
    """
    和原版的唯一区别：高度检查不再和固定的 0.25m 比，
    而是和“附近已经确认为地面的块”的高度比。
    从内往外处理，内圈先确认，外圈拿内圈当参照，所以能一路“爬”上土包。
    最内圈没有参照，用雷达安装高度推出的地面高度 0 作参照。
    """
    r = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0]) + np.pi
    ground = np.zeros(len(pts), bool)
    accepted = []                                   # 已确认地面块的中心 (x, y, z)
    for r0, r1, n_ring, n_sec in zones:
        ring = np.floor((r - r0) / ((r1 - r0) / n_ring)).astype(int)
        sec = np.floor(th / (2 * np.pi / n_sec)).astype(int)
        inzone = (r >= r0) & (r < r1)
        for i in range(n_ring):                     # 环从内到外
            new = []
            for j in range(n_sec):
                idx = np.nonzero(inzone & (ring == i) & (sec == j))[0]
                if len(idx) < min_pts:
                    continue
                p = pts[idx]
                lpr = np.sort(p[:, 2])[:num_lpr].mean()
                cand = p[:, 2] < lpr + th_seed
                for _ in range(iters):
                    if cand.sum() < 3:
                        break
                    normal, d, w = Z.fit_plane_pca(p[cand])
                    cand = np.abs(p @ normal + d) < th_dist
                if not cand.any():
                    continue
                c = p[cand].mean(0)
                # 参照高度：附近已确认地面块的中位数高度；没有就用 0
                if accepted:
                    A = np.array(accepted)
                    near = np.hypot(A[:, 0] - c[0], A[:, 1] - c[1]) < ref_radius
                    ref = np.median(A[near, 2]) if near.any() else 0.0
                else:
                    ref = 0.0
                ok = normal[2] > min_upright and w[0] < max_flat and abs(c[2] - ref) < max_jump
                if ok:
                    ground[idx[cand]] = True
                    new.append(c)
            accepted.extend(new)                    # 一整环处理完再加入，避免同环互相影响
    return ground


def zone_ground_adaptive_v2(pts, num_lpr=10, th_seed=0.10, th_dist=0.05, iters=3, min_pts=10,
                            min_upright=np.cos(np.radians(40)), max_flat=0.002,
                            max_jump=0.15, ref_radius=1.5, zones=Z.ZONES):
    """
    v1 的毛病：在坡上，1.5m 外的邻居本来就比我高/低几十厘米，直接比高度当然对不上。
    v2 改为：用邻居块的【平面】外推到我这里，预测“地面在这里应该多高”，再和我比。
    """
    r = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0]) + np.pi
    ground = np.zeros(len(pts), bool)
    accepted = []                                   # (cx, cy, nx, ny, nz, d)
    for r0, r1, n_ring, n_sec in zones:
        ring = np.floor((r - r0) / ((r1 - r0) / n_ring)).astype(int)
        sec = np.floor(th / (2 * np.pi / n_sec)).astype(int)
        inzone = (r >= r0) & (r < r1)
        for i in range(n_ring):
            new = []
            for j in range(n_sec):
                idx = np.nonzero(inzone & (ring == i) & (sec == j))[0]
                if len(idx) < min_pts:
                    continue
                p = pts[idx]
                lpr = np.sort(p[:, 2])[:num_lpr].mean()
                cand = p[:, 2] < lpr + th_seed
                for _ in range(iters):
                    if cand.sum() < 3:
                        break
                    normal, d, w = Z.fit_plane_pca(p[cand])
                    cand = np.abs(p @ normal + d) < th_dist
                if not cand.any():
                    continue
                c = p[cand].mean(0)
                if accepted:
                    A = np.array(accepted)
                    near = np.hypot(A[:, 0] - c[0], A[:, 1] - c[1]) < ref_radius
                    if near.any():
                        B = A[near]
                        # 邻居平面 n·x + d = 0 在 (c.x, c.y) 处的高度
                        z_pred = -(B[:, 2] * c[0] + B[:, 3] * c[1] + B[:, 5]) / B[:, 4]
                        ref = np.median(z_pred)
                    else:
                        ref = None                  # 没有邻居：跳过高度检查
                else:
                    ref = 0.0
                elev_ok = ref is None or abs(c[2] - ref) < max_jump
                ok = normal[2] > min_upright and w[0] < max_flat and elev_ok
                if ok:
                    ground[idx[cand]] = True
                    new.append([c[0], c[1], *normal, d])
            accepted.extend(new)
    return ground


print("\n====== 作业2：自适应高度检查 ======")
g = zone_ground_adaptive(P)
Z.evaluate("v1 比邻居高度        ", g, gt, 0)
g = zone_ground_adaptive_v2(P)
Z.evaluate("v2 用邻居平面外推    ", g, gt, 0)
g, _ = Z.zone_ground(P, max_elev=9)
Z.evaluate("对照：关掉高度检查    ", g, gt, 0)


# ---------- 作业3：机器人站在坡上，雷达整体倾斜 ----------
def rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


print("\n====== 作业3：雷达整体倾斜 25° ======")
R_tilt = rot_y(25)
P_tilt = P @ R_tilt.T                              # 雷达坐标系里看到的点云是歪的
g = zone_ground_adaptive_v2(P_tilt)
Z.evaluate("倾斜点云直接分割      ", g, gt, 0)
g = zone_ground_adaptive_v2(P_tilt @ rot_y(-25).T)
Z.evaluate("IMU完美校正          ", g, gt, 0)

# IMU 能测出重力方向，即知道倾斜角；把点云转回重力坐标系再分割
imu_err = 2.0                                      # 假设 IMU 姿态有 2° 误差
R_fix = rot_y(-(25 - imu_err))
g = zone_ground_adaptive_v2(P_tilt @ R_fix.T)
Z.evaluate("IMU校正后(残差2°)    ", g, gt, 0)
