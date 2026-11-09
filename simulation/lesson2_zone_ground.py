"""
第2课：起伏地形上的地面分割 —— 单平面 RANSAC vs 分区拟合（Patchwork 思想的简化版）

Patchwork / Patchwork++ 的三个核心想法（本文件各实现一个简化版）：
  1. 同心区域模型 CZM：以雷达为中心，按“距离环 × 角度扇区”把地面切成很多小块(bin)；
     近处点密，块切小；远处点稀，块切大。
  2. 分区平面拟合 R-GPF：每个块里只用“最低的一批点”做种子，用 PCA 拟合局部平面，迭代几次。
  3. 地面似然判断 GLE：拟合出的平面要满足
       - 竖直度 uprightness：法向量够竖直（允许坡，但不能是墙）
       - 平坦度 flatness：最小特征值够小（点确实贴着一个面）
       - 高度 elevation：近处块的平面不能比“地面该在的高度”高太多（防止把石头顶面当地面）
     不满足就整块判为非地面。
Patchwork++ 在此之上又加了：参数自适应(A-GLE)、时间回退(TGR)、竖直面拟合(R-VPF)、
反射噪声去除(RNR)。这些第2课不实现，面试时能说清它们解决什么问题即可。

运行：python3 lesson2_zone_ground.py   （需与 lesson1_pointcloud_basics.py 同目录）
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lesson1_pointcloud_basics as L

rng = np.random.default_rng(1)
GROUND, ROCK, TRUNK, TOY, NOISE = 0, 1, 2, 3, 4
NAMES = {ROCK: "石块25cm", TRUNK: "树干1.5m", TOY: "玩具12cm"}
SENSOR_H = 0.40          # 雷达装在离地 40cm 处；坐标原点 = 雷达所在位置正下方的地面


# ---------------------------------------------------------------
# 场景：起伏草坪（局部坡度最大约 25°）+ 一个土包 + 三个障碍物
# ---------------------------------------------------------------
def terrain(x, y):
    return (0.30 * np.sin(0.6 * x) + 0.20 * np.sin(0.8 * y + 1.0)
            + 0.40 * np.exp(-((x - 3.5) ** 2 + (y + 3.0) ** 2) / 2.0)) - 0.20 * np.sin(1.0)


def box(center, size, n):
    """在盒子侧面和顶面采样（底面贴地，看不到）"""
    p = rng.uniform(-0.5, 0.5, size=(n, 3)) * size
    face = rng.integers(0, 3, n)
    for ax in (0, 1):
        m = face == ax
        p[m, ax] = np.sign(p[m, ax]) * size[ax] / 2
    p[face == 2, 2] = size[2] / 2
    p[:, 2] += size[2] / 2
    return p + [center[0], center[1], terrain(*center)]


def make_scene():
    # 地面：r 均匀分布 → 点密度随距离 ~1/r，和真实雷达一样近密远疏
    n = 50000
    r = rng.uniform(0.3, 7.0, n)
    th = rng.uniform(-np.pi, np.pi, n)
    x, y = r * np.cos(th), r * np.sin(th)
    ground = np.column_stack([x, y, terrain(x, y) + rng.normal(0, 0.015, n)])

    rock = box((2.0, 1.0), np.array([0.4, 0.3, 0.25]), 1500)
    toy = box((-1.5, 2.5), np.array([0.3, 0.2, 0.12]), 800)
    t = rng.uniform(-np.pi, np.pi, 2000)
    h = rng.uniform(0, 1.5, 2000)
    trunk = np.column_stack([-3 + 0.12 * np.cos(t), -2 + 0.12 * np.sin(t), h + terrain(-3, -2)])
    noise = np.column_stack([rng.uniform(-6, 6, (200, 2)), rng.uniform(-0.5, 2.0, 200)])

    pts = np.vstack([ground, rock, trunk, toy, noise])
    gt = np.concatenate([np.full(len(a), c) for a, c in
                         [(ground, GROUND), (rock, ROCK), (trunk, TRUNK), (toy, TOY), (noise, NOISE)]])
    return pts, gt


# ---------------------------------------------------------------
# 分区地面拟合（简化版 Patchwork）
# ---------------------------------------------------------------
# 每个区域：(起始半径, 结束半径, 环数, 扇区数)。近处切得细，远处切得粗。
ZONES = [(0.3, 2.0, 4, 16), (2.0, 4.0, 4, 32), (4.0, 7.0, 3, 32)]


def fit_plane_pca(p):
    c = p.mean(0)
    cov = np.cov((p - c).T)
    w, v = np.linalg.eigh(cov)            # 特征值从小到大
    normal = v[:, 0]
    if normal[2] < 0:
        normal = -normal
    return normal, -normal @ c, w          # 平面 n·x + d = 0，以及三个特征值


def zone_ground(pts, num_lpr=10, th_seed=0.10, th_dist=0.05, iters=3,
                min_pts=10, min_upright=np.cos(np.radians(40)), max_flat=0.002, max_elev=0.25,
                zones=ZONES):
    r = np.hypot(pts[:, 0], pts[:, 1])
    th = np.arctan2(pts[:, 1], pts[:, 0]) + np.pi
    ground = np.zeros(len(pts), bool)
    bins = []                              # 记录每块的判断结果，画图用
    for r0, r1, n_ring, n_sec in zones:
        ring = np.floor((r - r0) / ((r1 - r0) / n_ring)).astype(int)
        sec = np.floor(th / (2 * np.pi / n_sec)).astype(int)
        inzone = (r >= r0) & (r < r1)
        for i in range(n_ring):
            for j in range(n_sec):
                idx = np.nonzero(inzone & (ring == i) & (sec == j))[0]
                if len(idx) < min_pts:
                    continue
                p = pts[idx]
                # R-GPF：用最低的 num_lpr 个点的平均高度定种子
                lpr = np.sort(p[:, 2])[:num_lpr].mean()
                cand = p[:, 2] < lpr + th_seed
                for _ in range(iters):
                    if cand.sum() < 3:
                        break
                    normal, d, w = fit_plane_pca(p[cand])
                    cand = np.abs(p @ normal + d) < th_dist
                # GLE：三个检查
                upright = normal[2] > min_upright
                flat = w[0] < max_flat
                center = p[cand].mean(0) if cand.any() else p.mean(0)
                # 高度检查只在近处做：近处地面应当在“雷达高度以下约 SENSOR_H”附近
                elev_ok = (r0 > 2.0) or (center[2] < max_elev)
                ok = upright and flat and elev_ok
                if ok:
                    ground[idx[cand]] = True
                bins.append((r0, r1, n_ring, n_sec, i, j, ok, len(idx)))
    return ground, bins


def evaluate(name, gmask, gt, ms):
    tp = (gmask & (gt == GROUND)).sum()
    P, R = tp / max(gmask.sum(), 1), tp / (gt == GROUND).sum()
    obs = "  ".join(f"{NAMES[c]}检出率 {(~gmask[gt == c]).mean():.2f}" for c in (ROCK, TRUNK, TOY))
    print(f"[{name}] {ms:6.1f} ms | 地面 P={P:.3f} R={R:.3f} | {obs}")
    return P, R


def main():
    pts, gt0 = make_scene()
    # 复用第1课：体素 + SOR
    p2, inv = L.voxel_downsample(pts, 0.05)
    g2 = np.array([np.bincount(gt0[inv == v], minlength=5).argmax() for v in range(len(p2))])
    keep = L.statistical_outlier_removal(p2)
    P, gt = p2[keep], g2[keep]
    print(f"预处理后 {len(P)} 点\n")

    res = {}
    for name, tilt in [("RANSAC 单平面(限15°)", 15), ("RANSAC 单平面(限40°)", 40)]:
        L.rng = np.random.default_rng(0)
        t = time.perf_counter()
        g, _ = L.ransac_ground(P, dist_thresh=0.05, iters=300, max_tilt_deg=tilt)
        res[name] = (g, evaluate(name, g, gt, 1000 * (time.perf_counter() - t)))

    t = time.perf_counter()
    gz, bins = zone_ground(P)
    res["分区拟合"] = (gz, evaluate("分区拟合            ", gz, gt, 1000 * (time.perf_counter() - t)))
    rej = [b for b in bins if not b[6]]
    print(f"\n分区拟合共 {len(bins)} 个有效块，判为非地面的块 {len(rej)} 个")

    # ---- 画图 ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    s = 0.4
    a = ax[0, 0]
    sc = a.scatter(P[:, 0], P[:, 1], s=s, c=P[:, 2], cmap="terrain")
    plt.colorbar(sc, ax=a, label="z (m)")
    a.set_title("terrain height (rolling lawn + mound)")
    for a, key in [(ax[0, 1], "RANSAC 单平面(限40°)"), (ax[1, 0], "分区拟合")]:
        g, (p_, r_) = res[key]
        miss = (~g) & (gt == GROUND)           # 地面被误判为障碍
        hit = (~g) & (gt != GROUND)            # 正确找到的障碍
        a.scatter(P[g, 0], P[g, 1], s=s, c="lightgray")
        a.scatter(P[miss, 0], P[miss, 1], s=2, c="tab:orange", label="ground judged obstacle")
        a.scatter(P[hit, 0], P[hit, 1], s=4, c="tab:red", label="true obstacle found")
        a.legend(loc="upper right", fontsize=8)
        a.set_title(("single-plane RANSAC" if "RANSAC" in key else "zone-wise fitting")
                    + f"   P={p_:.2f} R={r_:.2f}")
    # 画分区网格：绿色=判为地面的块，红色=拒绝的块
    a = ax[1, 1]
    for r0, r1, n_ring, n_sec, i, j, ok, n in bins:
        rr0 = r0 + i * (r1 - r0) / n_ring
        rr1 = rr0 + (r1 - r0) / n_ring
        tt = np.linspace(j, j + 1, 8) * 2 * np.pi / n_sec - np.pi
        xs = np.concatenate([rr0 * np.cos(tt), rr1 * np.cos(tt[::-1])])
        ys = np.concatenate([rr0 * np.sin(tt), rr1 * np.sin(tt[::-1])])
        a.fill(xs, ys, color="tab:green" if ok else "tab:red", alpha=0.35 if ok else 0.8, lw=0.3,
               edgecolor="k")
    for c, (x, y) in [(ROCK, (2, 1)), (TRUNK, (-3, -2)), (TOY, (-1.5, 2.5))]:
        a.plot(x, y, "k*", ms=10)
    a.set_title("zone grid (green = ground bin, red = rejected bin, * = obstacle)")
    for a in ax.ravel():
        a.set_aspect("equal")
        a.set_xlim(-7.3, 7.3)
        a.set_ylim(-7.3, 7.3)
    plt.tight_layout()
    plt.savefig("lesson2_result.png", dpi=105)
    print("图已保存: lesson2_result.png")


if __name__ == "__main__":
    main()
