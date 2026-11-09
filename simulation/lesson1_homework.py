"""
第1课作业：每次只改一个参数，和基线对比。
运行：python3 lesson1_homework.py   （需与 lesson1_pointcloud_basics.py 放在同一目录）
"""
import numpy as np
import lesson1_pointcloud_basics as L


def run(name, slope=0.03, leaf=0.05, dist_thresh=0.05, min_size=30):
    L.rng = np.random.default_rng(0)            # 每次重置随机种子，保证可比
    pts, gt = L.make_garden_scene(slope=slope)
    m = L.crop_box(pts, [-6, -6, -0.3], [6, 6, 2.0])
    p1, g1 = pts[m], gt[m]
    p2, inv = L.voxel_downsample(p1, leaf)
    g2 = np.array([np.bincount(g1[inv == v], minlength=4).argmax() for v in range(len(p2))])
    keep = L.statistical_outlier_removal(p2)
    p3, g3 = p2[keep], g2[keep]

    gmask, plane = L.ransac_ground(p3, dist_thresh=dist_thresh)
    if gmask is None:
        print(f"[{name}] 体素后 {len(p2)} 点；RANSAC 没找到任何合格平面！")
        return
    tp = (gmask & (g3 == L.GROUND)).sum()
    P, R = tp / gmask.sum(), tp / (g3 == L.GROUND).sum()

    obs = p3[~gmask]
    obs_gt = g3[~gmask]
    labels, keepc = L.euclidean_clustering(obs, tol=0.15, min_size=min_size)
    found = []
    for c in np.nonzero(keepc)[0]:
        sel = labels == c
        cp = obs[sel]
        major = np.bincount(obs_gt[sel], minlength=4).argmax()
        kind = {0: "地面误检", 1: "石块", 2: "树干", 3: "噪声"}[major]
        found.append(f"{kind}({sel.sum()}点,高{cp[:, 2].max() - cp[:, 2].min():.2f}m)")
    print(f"[{name}] 体素后{len(p2)}点 | 地面P={P:.3f} R={R:.3f} | "
          f"障碍物 {len(found)} 个: {', '.join(found) if found else '无'}")


print("坡度 0.15 对应角度:", round(np.degrees(np.arctan(0.15)), 1), "度\n")
run("基线           ")
run("作业1 阈值0.10 ", dist_thresh=0.10)
run("作业2 体素0.2  ", leaf=0.2)
run("作业2b 体素0.2 + min_size=5", leaf=0.2, min_size=5)
run("作业3 坡度0.15 ", slope=0.15)
run("作业3b 坡度0.30", slope=0.30)
