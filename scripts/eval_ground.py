"""
地面分割评测：单平面 RANSAC vs Patchwork++，在 SemanticKITTI 上算精确率/召回率/F1 和耗时（计划第 5–7 天）

数据目录（SemanticKITTI 标准结构）：
  <root>/sequences/08/velodyne/000000.bin   点云，每点 4 个 float32
  <root>/sequences/08/labels/000000.label   标签，每点 1 个 uint32，低 16 位是语义类别

用法：
  python3 scripts/eval_ground.py --root ~/data/semantickitti --seq 08 --frames 50 --out results/ground_08.csv

依赖：numpy open3d pypatchworkpp（Patchwork++ 仓库里 `make pyinstall` 安装）
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

# 视为“地面”的语义类别。Patchwork++ 论文的评测口径请自己核对后在 README 写明。
# 40 road, 44 parking, 48 sidewalk, 49 other-ground, 60 lane-marking, 72 terrain
GROUND_LABELS = {40, 44, 48, 49, 60, 72}
# 有争议的类别：植被(70)、未标注(0)。默认不计入评测，避免口径不清影响结论。
IGNORE_LABELS = {0, 1, 70}


def load_frame(seq_dir: Path, idx: int):
    pts = np.fromfile(seq_dir / "velodyne" / f"{idx:06d}.bin", dtype=np.float32).reshape(-1, 4)
    lab = np.fromfile(seq_dir / "labels" / f"{idx:06d}.label", dtype=np.uint32) & 0xFFFF
    return pts, lab


def ransac_ground(pts, dist=0.15, iters=200):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64)))
    _, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=iters)
    mask = np.zeros(len(pts), bool)
    mask[inliers] = True
    return mask


class PatchworkPP:
    def __init__(self):
        import pypatchworkpp
        self.params = pypatchworkpp.Parameters()
        self.pw = pypatchworkpp.patchworkpp(self.params)

    def __call__(self, pts):
        self.pw.estimateGround(pts)
        mask = np.zeros(len(pts), bool)
        if hasattr(self.pw, "getGroundIndices"):        # 新版本直接给下标
            mask[np.asarray(self.pw.getGroundIndices(), dtype=int)] = True
        else:                                           # 旧版本只给地面点坐标，用最近邻反查
            from scipy.spatial import cKDTree
            ground = np.asarray(self.pw.getGround())[:, :3]
            d, idx = cKDTree(pts[:, :3]).query(ground)
            mask[idx[d < 1e-4]] = True
        return mask


def metrics(pred, lab):
    valid = ~np.isin(lab, list(IGNORE_LABELS))
    gt = np.isin(lab, list(GROUND_LABELS))
    p, g = pred[valid], gt[valid]
    tp = (p & g).sum()
    prec = tp / max(p.sum(), 1)
    rec = tp / max(g.sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--seq", default="08")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--step", type=int, default=1, help="每隔几帧取一帧")
    ap.add_argument("--out", default="results/ground_eval.csv")
    args = ap.parse_args()

    seq_dir = Path(args.root).expanduser() / "sequences" / args.seq
    methods = {"ransac": ransac_ground, "patchworkpp": PatchworkPP()}
    rows = []
    for k in range(args.frames):
        idx = k * args.step
        if not (seq_dir / "velodyne" / f"{idx:06d}.bin").exists():
            break
        pts, lab = load_frame(seq_dir, idx)
        for name, fn in methods.items():
            t = time.perf_counter()
            pred = fn(pts)
            ms = (time.perf_counter() - t) * 1000
            prec, rec, f1 = metrics(pred, lab)
            rows.append({"frame": idx, "method": name, "precision": prec, "recall": rec, "f1": f1, "ms": ms})
        print(f"帧 {idx:06d} 完成")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n序列 {args.seq}，{len(rows) // len(methods)} 帧，结果写入 {args.out}")
    print(f"{'方法':12s} {'精确率':>8s} {'召回率':>8s} {'F1':>8s} {'耗时ms(中位)':>12s}")
    for name in methods:
        r = [x for x in rows if x["method"] == name]
        m = lambda k: np.mean([x[k] for x in r])
        print(f"{name:12s} {m('precision'):8.3f} {m('recall'):8.3f} {m('f1'):8.3f} "
              f"{np.median([x['ms'] for x in r]):12.1f}")


if __name__ == "__main__":
    main()
