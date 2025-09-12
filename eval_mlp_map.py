
import argparse, yaml
from pathlib import Path
import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="data.yaml 路径")
    # 保持向后兼容：--pred-root 仍可用（等价于 --pred-root-awe）
    ap.add_argument("--pred-root", type=str, default="", help="AWE 融合预测根目录（兼容旧参数名）")
    ap.add_argument("--pred-root-awe", type=str, default="", help="AWE 融合预测根目录（新）")
    ap.add_argument("--pred-root-base", type=str, default="", help="未增强（基线）预测根目录（可选）")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--iou-start", type=float, default=0.50)
    ap.add_argument("--iou-end",   type=float, default=0.95)
    ap.add_argument("--iou-step",  type=float, default=0.05)
    ap.add_argument("--show-class", action="store_true", help="打印 per-class AP 对比")
    return ap.parse_args()

# ---------- 读 YOLO 风格 txt ----------
def load_yolo_label(path: Path, is_pred: bool):
    if not path.exists():
        return np.zeros((0, 6 if is_pred else 5), dtype=np.float32)
    rows = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if is_pred:
                if len(parts) < 6:
                    parts = parts[:5] + ["0.0"]
                cls, cx, cy, w, h, conf = parts[:6]
                rows.append([float(cls), float(cx), float(cy), float(w), float(h), float(conf)])
            else:
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = parts[:5]
                rows.append([float(cls), float(cx), float(cy), float(w), float(h)])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6 if is_pred else 5), dtype=np.float32)

# ---------- 坐标与 IoU ----------
def xywhn_to_xyxy(arr):
    if arr.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = arr[:,1], arr[:,2], arr[:,3], arr[:,4]
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

def box_iou_matrix(A, B):
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    x11,y11,x12,y12 = A[:,0],A[:,1],A[:,2],A[:,3]
    x21,y21,x22,y22 = B[:,0],B[:,1],B[:,2],B[:,3]
    inter_x1 = np.maximum.outer(x11, x21)
    inter_y1 = np.maximum.outer(y11, y21)
    inter_x2 = np.minimum.outer(x12, x22)
    inter_y2 = np.minimum.outer(y12, y22)
    inter_w  = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h  = np.clip(inter_y2 - inter_y1, 0, None)
    inter    = inter_w * inter_h
    areaA    = (x12 - x11) * (y12 - y11)
    areaB    = (x22 - x21) * (y22 - y21)
    return inter / (areaA[:,None] + areaB[None,:] - inter + 1e-6)

# ---------- VOC-style AP ----------
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = max(mpre[i-1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])

def ap_per_class(all_preds, all_gts, nc, iou_thrs):
    # 组织 GT
    gts_by_img_cls = {}
    npos_by_cls = np.zeros(nc, dtype=np.int32)
    for stem, g in all_gts.items():
        if g.size == 0: continue
        for c in range(nc):
            gi = g[g[:,0]==c]
            if gi.size == 0: continue
            gts_by_img_cls[(stem,c)] = xywhn_to_xyxy(gi)
            npos_by_cls[c] += gi.shape[0]

    # 组织 Pred（按类聚合，并在类内按 conf 降序）
    preds_by_cls = [[] for _ in range(nc)]
    for stem, p in all_preds.items():
        if p.size == 0: continue
        for row in p:
            c, cx, cy, w, h, conf = row
            preds_by_cls[int(c)].append((stem, float(conf),
                                         np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2], np.float32)))
    for c in range(nc):
        preds_by_cls[c].sort(key=lambda x: -x[1])

    T = len(iou_thrs)
    aps_t = np.zeros((T, nc), dtype=np.float32)

    for ti, thr in enumerate(iou_thrs):
        matched = {}
        for (stem, cc), boxes in gts_by_img_cls.items():
            matched[(stem, cc)] = np.zeros(boxes.shape[0], dtype=bool)

        for c in range(nc):
            preds = preds_by_cls[c]
            if len(preds) == 0:
                aps_t[ti, c] = 0.0
                continue
            tp = np.zeros(len(preds), np.float32)
            fp = np.zeros(len(preds), np.float32)
            for i, (stem, conf, box) in enumerate(preds):
                key = (stem, c)
                gt_boxes = gts_by_img_cls.get(key, None)
                if gt_boxes is None:
                    fp[i] = 1.0
                    continue
                flags = matched[key]
                ious = box_iou_matrix(box[None,:], gt_boxes)[0]
                j = int(np.argmax(ious)); iou_best = ious[j]
                if iou_best >= thr and not flags[j]:
                    tp[i] = 1.0; flags[j] = True
                else:
                    fp[i] = 1.0

            tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
            npos = max(npos_by_cls[c], 1)
            recall = tp_c / npos
            precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
            aps_t[ti, c] = compute_ap(recall, precision)

    return aps_t  # [T,C]

# ---------- 你的 P/R 口径：单次贪心匹配 + 全局累计曲线取 F1 最大（IoU=0.5） ----------
def ultralytics_style_PR_like_yours(all_preds, all_gts, nc, iou_thr=0.5):
    total_gt = 0
    confs = []
    tps   = []

    gts_ic = {}
    for stem, g in all_gts.items():
        if g.size == 0: continue
        for c in range(nc):
            gi = g[g[:,0]==c]
            if gi.size == 0: continue
            gts_ic[(stem,c)] = xywhn_to_xyxy(gi)
            total_gt += gi.shape[0]

    preds_ic = {}
    for stem, p in all_preds.items():
        if p.size == 0: continue
        for row in p:
            c = int(row[0])
            preds_ic.setdefault((stem,c), []).append(row)

    # 注意：不做全局排序；每个 (img,cls) 内部按 conf 降序，再顺序追加
    for key in set(list(gts_ic.keys()) + list(preds_ic.keys())):
        gt_boxes = gts_ic.get(key, np.zeros((0,4), np.float32))
        pd_rows  = preds_ic.get(key, [])
        if len(pd_rows) == 0:
            continue
        pd = np.array(pd_rows, dtype=np.float32)  # [Np,6]
        order = np.argsort(-pd[:,5])
        pd = pd[order]
        pd_boxes = xywhn_to_xyxy(pd[:, :5])
        used = np.zeros((gt_boxes.shape[0],), dtype=bool)
        for i in range(pd.shape[0]):
            conf = float(pd[i,5])
            if gt_boxes.shape[0] == 0:
                confs.append(conf); tps.append(0); continue
            ious = box_iou_matrix(pd_boxes[i:i+1], gt_boxes)[0]
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not used[j]:
                used[j] = True; confs.append(conf); tps.append(1)
            else:
                confs.append(conf); tps.append(0)

    if len(confs) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0

    confs = np.array(confs, np.float32)
    tps   = np.array(tps,   np.float32)

    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(1.0 - tps)
    precision_curve = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    recall_curve    = tp_cum / max(total_gt, 1e-9)
    f1_curve        = 2 * precision_curve * recall_curve / np.maximum(precision_curve + recall_curve, 1e-9)
    i_best = int(np.argmax(f1_curve))
    P = float(precision_curve[i_best])
    R = float(recall_curve[i_best])
    F1 = float(f1_curve[i_best])
    return P, R, F1

# ---------- 单套预测评估 ----------
def evaluate_one(data, split, pred_root):
    names = data.get("names", [])
    nc = int(data.get("nc", len(names))) if names else int(data.get("nc", 1))

    rel = data.get(split)
    img_dir = (Path(data["__path__"]).parent / rel).resolve()
    gt_dir  = (img_dir.parent / "labels").resolve()
    pd_dir  = (Path(pred_root) / split / "labels").resolve()

    stems = [p.stem for p in img_dir.iterdir() if p.suffix.lower() in EXTS]
    all_gts, all_preds = {}, {}
    miss_pred = 0
    for s in stems:
        gt = load_yolo_label(gt_dir / f"{s}.txt", is_pred=False)
        pd = load_yolo_label(pd_dir / f"{s}.txt", is_pred=True)
        all_gts[s] = gt
        all_preds[s] = pd
        if pd.size == 0:
            miss_pred += 1

    iou_thrs = np.arange(0.50, 0.95 + 1e-9, 0.05)
    aps_t = ap_per_class(all_preds, all_gts, nc, iou_thrs)

    # 容忍浮点
    def _row_at(thrs, target, tol=1e-6):
        idx = int(np.argmin(np.abs(thrs - target)))
        return idx if abs(float(thrs[idx]) - float(target)) <= tol else None
    idx50 = _row_at(iou_thrs, 0.50)
    idx75 = _row_at(iou_thrs, 0.75)

    ap50   = float(np.nanmean(aps_t[idx50, :])) if idx50 is not None else float('nan')
    ap75   = float(np.nanmean(aps_t[idx75, :])) if idx75 is not None else float('nan')
    map5095= float(np.nanmean(aps_t)) if aps_t.size > 0 else 0.0

    P, R, F1 = ultralytics_style_PR_like_yours(all_preds, all_gts, nc, iou_thr=0.5)

    ap50_per_class   = aps_t[idx50, :] if idx50 is not None else np.zeros((nc,), dtype=np.float32)
    ap5095_per_class = aps_t.mean(axis=0)

    return {
        "stems": stems,
        "miss_pred": miss_pred,
        "ap50": ap50,
        "ap75": ap75,
        "map5095": map5095,
        "P": P, "R": R, "F1": F1,
        "aps_t": aps_t,
        "ap50_per_class": ap50_per_class,
        "ap5095_per_class": ap5095_per_class,
        "iou_thrs": iou_thrs,
        "nc": nc,
        "names": names if names and len(names)==int(nc) else [f"class_{i}" for i in range(int(nc))]
    }

def main():
    args = parse_args()
    data = yaml.safe_load(open(args.data, "r"))
    data["__path__"] = args.data  # 用于定位相对路径

    # 解析 AWE 预测目录
    pred_root_awe = args.pred_root_awe or args.pred_root
    if not pred_root_awe:
        raise SystemExit("[ERR] 请提供 --pred-root-awe（或旧参数 --pred-root）作为 AWE 融合预测目录")

    # 评估 AWE
    awe = evaluate_one(data, args.split, pred_root_awe)

    print(f"\n=== Evaluation on {args.split} ===")
    print("AWE Pred root :", str(Path(pred_root_awe).resolve()))
    print(f"IoU thresholds: {np.round(awe['iou_thrs'], 2)}")
    print(f"Images: {len(awe['stems'])}  |  missing preds (AWE): {awe['miss_pred']}")
    print(f"[AWE] AP@0.50 : {awe['ap50']:.4f}")
    print(f"[AWE] AP@0.75 : {awe['ap75']:.4f}")
    print(f"[AWE] AP@[.5:.95] (mAP): {awe['map5095']:.4f}")
    print(f"[AWE] P/R/F1 (IoU=0.5, YOUR LOGIC): {awe['P']:.4f} / {awe['R']:.4f} / {awe['F1']:.4f}")

    # 可选：评估基线并对比
    if args.pred_root_base:
        base = evaluate_one(data, args.split, args.pred_root_base)
        print("\nBaseline Pred root:", str(Path(args.pred_root_base).resolve()))
        print(f"Images: {len(base['stems'])}  |  missing preds (BASE): {base['miss_pred']}")
        print(f"[BASE] AP@0.50 : {base['ap50']:.4f}")
        print(f"[BASE] AP@0.75 : {base['ap75']:.4f}")
        print(f"[BASE] AP@[.5:.95] (mAP): {base['map5095']:.4f}")
        print(f"[BASE] P/R/F1 (IoU=0.5, YOUR LOGIC): {base['P']:.4f} / {base['R']:.4f} / {base['F1']:.4f}")

        # 汇总对比
        def fmt_delta(a, b):
            d = a - b
            r = (d / (b + 1e-12)) * 100.0
            return f"{a:.4f}  (Δ={d:+.4f}, {r:+.1f}%)"

        print("\n=== AWE vs. BASE (higher is better) ===")
        print("AP@0.50     :", fmt_delta(awe["ap50"],    base["ap50"]))
        print("AP@0.75     :", fmt_delta(awe["ap75"],    base["ap75"]))
        print("mAP@[.5:.95]:", fmt_delta(awe["map5095"], base["map5095"]))
        print("Precision   :", fmt_delta(awe["P"],       base["P"]))
        print("Recall      :", fmt_delta(awe["R"],       base["R"]))
        print("F1          :", fmt_delta(awe["F1"],      base["F1"]))

        # 每类 AP 对比
        if args.show_class:
            names = awe["names"]
            print("\nPer-class AP@50:95 (AWE vs BASE, Δ)")
            for i, name in enumerate(names):
                awe_c  = awe["ap5095_per_class"][i]
                base_c = base["ap5095_per_class"][i]
                d = awe_c - base_c
                r = (d / (base_c + 1e-12)) * 100.0
                print(f"{name:>12s} : {awe_c:.4f} vs {base_c:.4f}  (Δ={d:+.4f}, {r:+.1f}%)")
    else:
        # 只评估了 AWE
        names = awe["names"]
        if args.show_class:
            print("\nPer-class AP:")
            ap50_c   = awe["ap50_per_class"]
            ap5095_c = awe["ap5095_per_class"]
            for i, name in enumerate(names):
                print(f"{name:>12s} | AP50={ap50_c[i]:.4f}  AP50:95={ap5095_c[i]:.4f}")

if __name__ == "__main__":
    main()
