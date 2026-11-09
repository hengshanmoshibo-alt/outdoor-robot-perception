"""
第4课作业讲解
  作业1：算一算误差在图像上移动多少像素，和物体本身占多少像素比
  作业2：用“框大小 → 距离先验”选深度簇，并加一致性检查
运行：python3 lesson4_homework.py   （与第1、2、4课文件同目录）
"""
import numpy as np
import lesson4_fusion as F
import lesson2_zone_ground as Z

pts, lab = F.make_scene()
R, t = F.camera_pose()
nonground = ~Z.zone_ground(pts, max_elev=9)[0]
F.rng = np.random.default_rng(4)
boxes = F.fake_detector(pts, lab, R, t)

# ---------------- 作业1 ----------------
print("== 作业1：误差移动的像素 vs 物体在图上的大小 ==")
fx = F.K[0, 0]
shift = {d: fx * np.tan(np.radians(d)) for d in (1, 3)}
print(f"  角度误差导致的像素平移：1° ≈ {shift[1]:.0f}px，3° ≈ {shift[3]:.0f}px（图像中心附近）")
for name, i, (x0, y0, x1, y1) in boxes:
    w, h = x1 - x0, y1 - y0
    print(f"  {name:8s} 框宽 {w:5.0f}px 框高 {h:5.0f}px | 3°平移/宽 = {shift[3] / w:.2f}  "
          f"3°平移/高 = {shift[3] / h:.2f}")

# ---------------- 作业2 ----------------
KNOWN_WIDTH = {"rock": 0.4, "toy": 0.3, "box_far": 0.5}   # 检测类别 → 典型宽度(m)，有误差也没关系


def depth_clusters(z, gap=0.3, min_pts=10):
    """一维聚类：深度排序后，相邻差 > gap 就切开"""
    order = np.argsort(z)
    zs = z[order]
    cuts = np.nonzero(np.diff(zs) > gap)[0] + 1
    groups = np.split(order, cuts)
    return [g for g in groups if len(g) >= min_pts]


def fuse_with_prior(box, name, pts, R, t, use_prior=True, tol=0.35):
    uv, z, valid = F.project(pts, R, t)
    x0, y0, x1, y1 = box
    inbox = valid & (uv[:, 0] >= x0) & (uv[:, 0] <= x1) & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    idx = np.nonzero(inbox & nonground)[0]
    if len(idx) < 10:
        idx = np.nonzero(inbox)[0]
    groups = depth_clusters(z[idx])
    if not groups:
        return None, "框内没有足够的点"
    z_prior = fx * KNOWN_WIDTH[name] / (x1 - x0)          # 针孔模型：宽度像素 = fx·真实宽度/深度
    meds = [np.median(z[idx[g]]) for g in groups]
    if not use_prior:
        k = int(np.argmin(meds))                            # 第4课的做法：取最近一簇
    else:
        k = int(np.argmin([abs(m - z_prior) for m in meds]))
        if abs(meds[k] - z_prior) / z_prior > tol:          # 一致性检查：差太多就不给结果
            return None, f"不可信(先验{z_prior:.1f}m，最接近的簇{meds[k]:.1f}m)"
    sel = idx[groups[k]]
    return np.median(pts[sel, :2], axis=0), f"先验{z_prior:.1f}m 选中簇{meds[k]:.1f}m"


print("\n== 作业2：框大小先验 + 一致性检查 ==")
for title, kw in [("外参准确", {}), ("偏航3°", {"yaw_err": 3}), ("偏航6°", {"yaw_err": 6}),
                  ("俯仰3°", {"pitch_err": 3})]:
    Re, te = F.camera_pose(**kw)
    print(f"  [{title}]")
    for name, i, box in boxes:
        gt = np.array(F.OBJECTS[i][1])
        res = []
        for use in (False, True):
            est, info = fuse_with_prior(box, name, pts, Re, te, use_prior=use)
            err = f"{np.linalg.norm(est - gt):.2f}m" if est is not None else "—"
            res.append(f"{'先验法' if use else '最近簇'} 误差 {err:>6s} ({info})")
        print(f"    {name:8s} " + " | ".join(res))
