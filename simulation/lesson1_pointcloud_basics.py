"""
第1课：点云基础 + 四种常用处理（对应 PCL 里的类）

  步骤                   本文件函数                 PCL 对应类
  1. 范围裁剪             crop_box                  pcl::CropBox / pcl::PassThrough
  2. 体素降采样           voxel_downsample          pcl::VoxelGrid
  3. 统计离群点去除       statistical_outlier_removal  pcl::StatisticalOutlierRemoval
  4. 地面分割(RANSAC平面) ransac_ground             pcl::SACSegmentation (SACMODEL_PLANE)
  5. 欧式聚类找障碍物     euclidean_clustering      pcl::EuclideanClusterExtraction

数据：程序自己生成一个“庭院”场景（草地 + 石块 + 树干 + 噪声），
      这样你能知道每个点的真实类别，后面可以算精确率/召回率。
运行：python3 lesson1_pointcloud_basics.py
"""
import time
import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# 真实标签：0=地面 1=石块 2=树干 3=噪声
GROUND, ROCK, TRUNK, NOISE = 0, 1, 2, 3


# ---------------------------------------------------------------
# 0. 造数据：点云就是一个 N×3 的数组，每行是 (x, y, z)，单位米
# ---------------------------------------------------------------
def make_garden_scene(slope=0.03):
    # 草地：12m×12m，略有坡度(x方向每米升高 slope 米)，表面起伏 ±2cm
    n = 60000
    xy = rng.uniform(-6, 6, size=(n, 2))
    z = slope * xy[:, 0] + rng.normal(0, 0.02, n)
    ground = np.column_stack([xy, z])

    # 石块：中心(2,1)，0.4×0.3×0.25m 的盒子，只采表面点
    rock = rng.uniform([-0.2, -0.15, 0], [0.2, 0.15, 0.25], size=(3000, 3))
    face = rng.integers(0, 3, len(rock))           # 随机把点推到某个面上
    for axis, half in [(0, 0.2), (1, 0.15)]:
        m = face == axis
        rock[m, axis] = np.sign(rock[m, axis]) * half
    rock[face == 2, 2] = 0.25
    rock += [2.0, 1.0, slope * 2.0]

    # 树干：中心(-3,-2)，半径0.12m，高1.5m 的圆柱侧面
    theta = rng.uniform(0, 2 * np.pi, 4000)
    h = rng.uniform(0, 1.5, 4000)
    trunk = np.column_stack([-3 + 0.12 * np.cos(theta),
                             -2 + 0.12 * np.sin(theta),
                             h + slope * -3])

    # 噪声：激光雷达常见的零散飞点
    noise = rng.uniform([-6, -6, -0.5], [6, 6, 2.0], size=(400, 3))

    # 远处点：超出关心范围的点（比如邻居家的墙）
    far = rng.uniform([8, -6, 0], [10, 6, 2], size=(2000, 3))

    pts = np.vstack([ground, rock, trunk, noise, far])
    labels = np.concatenate([np.full(len(ground), GROUND), np.full(len(rock), ROCK),
                             np.full(len(trunk), TRUNK), np.full(len(noise), NOISE),
                             np.full(len(far), NOISE)])
    return pts, labels


# ---------------------------------------------------------------
# 1. 范围裁剪：只保留机器人附近关心的区域，最便宜也最先做
# ---------------------------------------------------------------
def crop_box(pts, lo, hi):
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    return mask


# ---------------------------------------------------------------
# 2. 体素降采样：把空间切成边长 leaf 的小立方体，每个格子里的点
#    用它们的质心代替。点数大降，形状基本保留。
# ---------------------------------------------------------------
def voxel_downsample(pts, leaf):
    idx = np.floor(pts / leaf).astype(np.int64)            # 每个点落在哪个格子
    _, inverse, counts = np.unique(idx, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    sums = np.zeros((len(counts), 3))
    np.add.at(sums, inverse, pts)                          # 按格子求和
    centroids = sums / counts[:, None]
    return centroids, inverse                               # inverse: 原始点→格子编号


# ---------------------------------------------------------------
# 3. 统计离群点去除(SOR)：算每个点到 k 个最近邻的平均距离 d，
#    d 超过 全体均值 + std_mul×标准差 的点视为离群点。
# ---------------------------------------------------------------
def statistical_outlier_removal(pts, k=16, std_mul=1.0):
    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=k + 1)                     # 第1个邻居是自己，去掉
    mean_d = dist[:, 1:].mean(axis=1)
    thresh = mean_d.mean() + std_mul * mean_d.std()
    return mean_d <= thresh


# ---------------------------------------------------------------
# 4. RANSAC 地面平面：反复随机取3点拟合平面，数“离平面足够近”的点，
#    保留内点最多的那个平面。只接受法向接近竖直的平面（避免把墙当地面）。
# ---------------------------------------------------------------
def ransac_ground(pts, dist_thresh=0.05, iters=200, max_tilt_deg=15):
    best_inliers, best_plane = None, None
    cos_limit = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        p1, p2, p3 = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        if abs(normal[2]) < cos_limit:                    # 太斜，不像地面
            continue
        d = -normal @ p1
        inliers = np.abs(pts @ normal + d) < dist_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_plane = inliers, (normal, d)
    return best_inliers, best_plane


# ---------------------------------------------------------------
# 5. 欧式聚类：距离小于 tol 的点算“连在一起”，用 BFS 找连通块。
#    点太少的块丢掉（多半是噪声）。
# ---------------------------------------------------------------
def euclidean_clustering(pts, tol=0.15, min_size=30):
    tree = cKDTree(pts)
    labels = np.full(len(pts), -1)
    cid = 0
    for i in range(len(pts)):
        if labels[i] != -1:
            continue
        queue = [i]
        labels[i] = cid
        while queue:
            j = queue.pop()
            for nb in tree.query_ball_point(pts[j], tol):
                if labels[nb] == -1:
                    labels[nb] = cid
                    queue.append(nb)
        cid += 1
    sizes = np.bincount(labels)
    keep = sizes >= min_size
    return labels, keep


def timed(name, fn, *a, **kw):
    t = time.perf_counter()
    out = fn(*a, **kw)
    print(f"  耗时 {1000 * (time.perf_counter() - t):7.1f} ms   [{name}]")
    return out


def main():
    pts, gt = make_garden_scene()
    print(f"原始点数: {len(pts)}")

    # 1. 裁剪：只要机器人周围 ±6m、高度 -0.3~2m
    m = timed("crop", crop_box, pts, lo=[-6, -6, -0.3], hi=[6, 6, 2.0])
    p1, g1 = pts[m], gt[m]
    print(f"1 裁剪后:      {len(p1):6d} 点")

    # 2. 体素降采样：5cm 格子
    p2, inv = timed("voxel", voxel_downsample, p1, leaf=0.05)
    # 给每个体素一个标签：取格子里多数点的真实类别（只为了后面评估）
    g2 = np.array([np.bincount(g1[inv == v], minlength=4).argmax() for v in range(len(p2))])
    print(f"2 体素降采样后: {len(p2):6d} 点")

    # 3. SOR 去离群点
    keep = timed("sor", statistical_outlier_removal, p2, k=16, std_mul=1.0)
    p3, g3 = p2[keep], g2[keep]
    removed_noise = ((~keep) & (g2 == NOISE)).sum()
    print(f"3 SOR后:       {len(p3):6d} 点  (去掉 {(~keep).sum()} 点，其中真噪声 "
          f"{removed_noise}/{(g2 == NOISE).sum()})")

    # 4. 地面分割 + 评估
    ground_mask, (normal, d) = timed("ransac", ransac_ground, p3)
    tp = (ground_mask & (g3 == GROUND)).sum()
    precision = tp / ground_mask.sum()
    recall = tp / (g3 == GROUND).sum()
    print(f"4 地面点 {ground_mask.sum()}，非地面点 {(~ground_mask).sum()}")
    print(f"  地面平面法向 {np.round(normal, 3)}  精确率 {precision:.3f}  召回率 {recall:.3f}")

    # 5. 对非地面点聚类
    obs = p3[~ground_mask]
    labels, keep_c = timed("cluster", euclidean_clustering, obs, tol=0.15, min_size=30)
    valid = [c for c in range(len(keep_c)) if keep_c[c]]
    print(f"5 找到障碍物簇 {len(valid)} 个：")
    boxes = []
    for c in valid:
        cp = obs[labels == c]
        lo, hi = cp.min(0), cp.max(0)
        boxes.append((lo, hi))
        print(f"  簇{c}: {len(cp):4d} 点  中心 {np.round((lo + hi) / 2, 2)}  尺寸 {np.round(hi - lo, 2)}")

    # 画图：俯视图 4 张
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    s = 0.3
    ax[0, 0].scatter(pts[:, 0], pts[:, 1], s=s, c=pts[:, 2], cmap="viridis")
    ax[0, 0].set_title(f"0 raw: {len(pts)} pts (color = height)")
    ax[0, 1].scatter(p2[:, 0], p2[:, 1], s=s, c=p2[:, 2], cmap="viridis")
    ax[0, 1].scatter(p2[~keep, 0], p2[~keep, 1], s=8, c="red", label="removed by SOR")
    ax[0, 1].legend(loc="upper right")
    ax[0, 1].set_title(f"1-3 crop+voxel+SOR: {len(p3)} pts")
    ax[1, 0].scatter(p3[ground_mask, 0], p3[ground_mask, 1], s=s, c="tab:green", label="ground")
    ax[1, 0].scatter(p3[~ground_mask, 0], p3[~ground_mask, 1], s=3, c="tab:red", label="non-ground")
    ax[1, 0].legend(loc="upper right")
    ax[1, 0].set_title(f"4 RANSAC ground  P={precision:.2f} R={recall:.2f}")
    ax[1, 1].scatter(p3[ground_mask, 0], p3[ground_mask, 1], s=s, c="lightgray")
    for c, (lo, hi) in zip(valid, boxes):
        cp = obs[labels == c]
        ax[1, 1].scatter(cp[:, 0], cp[:, 1], s=4)
        ax[1, 1].add_patch(plt.Rectangle(lo[:2], *(hi - lo)[:2], fill=False, lw=1.5))
    ax[1, 1].set_title(f"5 clusters -> {len(valid)} obstacles")
    for a in ax.ravel():
        a.set_aspect("equal")
        a.set_xlim(-6.5, 10.5)
        a.set_ylim(-6.5, 6.5)
    plt.tight_layout()
    plt.savefig("lesson1_result.png", dpi=110)
    print("图已保存: lesson1_result.png")


if __name__ == "__main__":
    main()
