import os
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from segment_anything import sam_model_registry, SamPredictor

# ---------------------------
# 路径配置（你的本地权重）
# ---------------------------
OWLV2_DIR = "/home/xuhq/project/owlv2/owlv2-base-patch16/"
SAM_CKPT = "/home/xuhq/project/clipsam/DFN5B-CLIP-ViT-H-14-378/sam_l.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 你指定的颜色（RGB）
colors_rgb = [
    [0, 191, 255],  # 深天蓝
    [255, 105, 180],  # 粉色
    [255, 215, 0],  # 金色
    [50, 205, 50],  # 酸橙绿
    [148, 0, 211]  # 紫色
]


# ---------------------------
# OWLv2: 文本 -> 框
# ---------------------------
def load_owlv2(model_dir: str):
    processor = Owlv2Processor.from_pretrained(model_dir)
    model = Owlv2ForObjectDetection.from_pretrained(model_dir).to(DEVICE).eval()
    return model, processor


@torch.no_grad()
def owlv2_detect(model, processor, image_pil: Image.Image, text_queries,
                 score_thr=0.1):
    """
    返回：
      boxes: [N,4] x1,y1,x2,y2 (float)
      scores:[N]
      labels:[N]
    """
    inputs = processor(text=text_queries, images=image_pil, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]]).to(DEVICE)  # (H,W)
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=score_thr
    )[0]

    boxes = results["boxes"].detach().cpu().numpy()
    scores = results["scores"].detach().cpu().numpy()
    labels = results["labels"].detach().cpu().numpy()
    return boxes, scores, labels


def nms_xyxy(boxes, scores, iou_thr=0.5):
    """简单 NMS，避免重复框"""
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep


def pick_topk_boxes(boxes, scores, labels, num_instances=2, iou_thr=0.5):
    """NMS 后选 top-k"""
    keep = nms_xyxy(boxes, scores, iou_thr=iou_thr)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

    order = scores.argsort()[::-1][:num_instances]
    return boxes[order], scores[order], labels[order]


# ---------------------------
# 改进的heatmap生成方法
# ---------------------------
def boxes_to_heatmap_v2(image_hw, boxes_xyxy, scores=None, sigma_ratio=0.15):
    """
    生成更自然、不规则的heatmap
    在box内生成多个高斯峰值点,模拟真实的注意力分布
    """
    H, W = image_hw
    heatmap = np.zeros((H, W), dtype=np.float32)

    if scores is None:
        scores = np.ones((len(boxes_xyxy),), dtype=np.float32)

    for box, s in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # 在box内生成多个高斯峰值点(根据box大小自适应数量)
        num_peaks = max(3, int(np.sqrt(bw * bh) / 20))

        for _ in range(num_peaks):
            # 随机采样峰值位置(偏向中心区域)
            cx = np.random.normal((x1 + x2) / 2, bw * 0.15)
            cy = np.random.normal((y1 + y2) / 2, bh * 0.15)
            cx = np.clip(cx, x1, x2)
            cy = np.clip(cy, y1, y2)

            # 变化的sigma(增加随机性)
            sigma = min(bw, bh) * sigma_ratio * np.random.uniform(0.8, 1.2)

            # 生成局部高斯(只计算3sigma范围内)
            y_range = slice(max(0, int(cy - 3 * sigma)), min(H, int(cy + 3 * sigma)))
            x_range = slice(max(0, int(cx - 3 * sigma)), min(W, int(cx + 3 * sigma)))

            yy, xx = np.ogrid[y_range, x_range]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

            heatmap[y_range, x_range] += s * gaussian / num_peaks

    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def boxes_to_heatmap_edge_based(image_np, boxes_xyxy, scores=None,
                                blur_size=51, edge_weight=2.0):
    """
    基于图像边缘信息生成heatmap,更贴合物体形状
    结合Canny边缘检测,让热图沿着物体轮廓分布
    """
    H, W = image_np.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)

    # 计算边缘强度
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = edges.astype(np.float32) / 255.0

    if scores is None:
        scores = np.ones((len(boxes_xyxy),), dtype=np.float32)

    for box, s in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        # 创建box mask
        mask = np.zeros((H, W), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0

        # 结合边缘信息(边缘处响应更强)
        edge_mask = mask * (1.0 + edge_weight * edges)

        # 高斯模糊平滑
        if blur_size % 2 == 0:
            blur_size += 1
        blurred = cv2.GaussianBlur(edge_mask, (blur_size, blur_size), 0)
        heatmap += s * blurred

    # 归一化
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


def enhance_heatmap(hm, clip_percentile=60, gamma=2.5):
    """
    增强heatmap对比度
    clip_percentile: 裁剪低响应值(背景变暗)
    gamma: 幂次增强,让高响应区域更突出
    """
    hm = hm.copy()

    # 1) 裁剪背景
    thr = np.percentile(hm, clip_percentile)
    hm = np.clip(hm - thr, 0, None)

    # 2) 重新归一化
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())

    # 3) 幂次增强
    hm = hm ** gamma
    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())

    return hm


def points_from_heatmap_and_boxes(heatmap, boxes_xyxy):
    """
    每个 box 内在 heatmap 上取最大响应点
    返回 points: [K,2] (x,y)
    """
    points = []
    for box in boxes_xyxy:
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(heatmap.shape[1] - 1, x2), min(heatmap.shape[0] - 1, y2)

        roi = heatmap[y1:y2, x1:x2]
        if roi.size == 0:
            # 兜底：box center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            points.append((cx, cy))
            continue

        idx = np.argmax(roi.reshape(-1))
        ry = idx // roi.shape[1]
        rx = idx % roi.shape[1]
        points.append((x1 + rx, y1 + ry))

    return np.array(points, dtype=np.int32)


# ---------------------------
# SAM：点 -> 多 mask
# ---------------------------
def load_sam(ckpt_path: str, model_type="vit_l"):
    sam = sam_model_registry[model_type](checkpoint=ckpt_path).to(DEVICE)
    predictor = SamPredictor(sam)
    return predictor


@torch.no_grad()
def sam_masks_from_points(predictor: SamPredictor, image_np: np.ndarray, points_xy: np.ndarray):
    """
    对每个点单独预测一个 mask
    points_xy: [K,2] (x,y)
    返回 masks: [K,H,W], scores: [K]
    """
    predictor.set_image(image_np)

    masks = []
    scores_out = []
    for p in points_xy:
        p = p.reshape(1, 2).astype(np.float32)
        labels = np.array([1], dtype=np.int32)

        m, s, _ = predictor.predict(
            point_coords=p,
            point_labels=labels,
            multimask_output=True
        )
        best = np.argmax(s)
        masks.append(m[best])
        scores_out.append(float(s[best]))

    return np.stack(masks, axis=0), np.array(scores_out)


# ---------------------------
# 改进的可视化函数
# ---------------------------
def visualize_heatmap_points_enhanced(image_np, heatmap, points,
                                      save_path="owlv2_heatmap_points_enhanced.png",
                                      alpha=0.7, cmap='hot'):
    """
    三合一对比图:原图、纯热图、叠加图
    适合展示算法效果
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 子图1: 原图
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # 子图2: 纯heatmap
    im = axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 子图3: 叠加图
    axes[2].imshow(image_np)
    axes[2].imshow(heatmap, cmap=cmap, alpha=alpha)
    for i, (x, y) in enumerate(points):
        axes[2].scatter([x], [y], s=150, c='cyan', marker='*',
                        edgecolors='white', linewidths=2, label=f'Point {i + 1}')
    axes[2].set_title('Overlay with Keypoints', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    if len(points) <= 5:
        axes[2].legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved enhanced visualization to: {save_path}")
    plt.close()


def visualize_for_paper(image_np, heatmap, points=None,
                        save_path="heatmap_for_paper.png",
                        alpha=0.65, cmap='hot', dpi=300, show_colorbar=True):
    """
    论文专用单图版本
    简洁、高清、适合出版
    """
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(image_np)
    im = plt.imshow(heatmap, cmap=cmap, alpha=alpha)

    if points is not None:
        plt.scatter(points[:, 0], points[:, 1],
                    s=200, c='cyan', marker='*',
                    edgecolors='white', linewidths=2.5,
                    label='Detected Keypoints', zorder=10)
        # plt.legend(loc='upper right', fontsize=12, framealpha=0.9)

    if show_colorbar:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label('Attention Intensity', rotation=270, labelpad=20, fontsize=11)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    print(f"[OK] Saved paper-ready figure to: {save_path}")
    plt.close()


def visualize_multimask(image_np, masks, points=None,
                        save_path="owlv2_sam_multimask.png"):
    """
    所有 mask 叠在同一张图里，每个 mask 用不同颜色
    """
    H, W, _ = image_np.shape
    plt.figure(figsize=(10, 8))
    plt.imshow(image_np)

    for i, mask in enumerate(masks):
        color = np.array(colors_rgb[i % len(colors_rgb)]) / 255.0
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        overlay[mask > 0] = color
        plt.imshow(overlay, alpha=0.45)

    if points is not None:
        for (x, y) in points:
            plt.scatter([x], [y], s=80, marker="o", c="white",
                        edgecolors='black', linewidths=1.5)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved multi-mask overlay to: {save_path}")
    plt.close()


# ---------------------------
# 主流程
# ---------------------------
def main():
    # ====== 配置参数 ======
    image_path = "/home/xuhq/project/siglip/0a0bc8ead03f292f_jpg.rf.0a0796e9988923027a7769786f4d5a7e.jpg"
    text_queries = ["kangaroo"]
    num_instances = 2
    score_thr = 0.10
    iou_thr = 0.50

    # 选择heatmap生成方法: 'v2' 或 'edge_based'
    heatmap_method = 'v2'  # 推荐用v2,更自然

    # ===========================

    assert os.path.exists(OWLV2_DIR), f"OWLv2 dir not found: {OWLV2_DIR}"
    assert os.path.exists(SAM_CKPT), f"SAM ckpt not found: {SAM_CKPT}"
    assert os.path.exists(image_path), f"Image not found: {image_path}"

    # 1) 加载模型
    print("[INFO] Loading OWLv2 model...")
    owlv2_model, owlv2_proc = load_owlv2(OWLV2_DIR)
    print("[INFO] Loading SAM model...")
    sam_predictor = load_sam(SAM_CKPT, model_type="vit_l")

    # 2) 加载图片
    image_pil = Image.open(image_path).convert("RGB")
    image_np = np.array(image_pil)
    H, W = image_np.shape[:2]
    print(f"[INFO] Image loaded: {W}x{H}")

    # 3) OWLv2 检测
    print(f"[INFO] Running OWLv2 detection with queries: {text_queries}")
    boxes, scores, labels = owlv2_detect(
        owlv2_model, owlv2_proc, image_pil, text_queries, score_thr=score_thr
    )

    if len(boxes) == 0:
        print("[WARN] OWLv2 found no boxes. Try lower score_thr or change prompt.")
        return

    # 4) NMS + top-k
    top_boxes, top_scores, top_labels = pick_topk_boxes(
        boxes, scores, labels, num_instances=num_instances, iou_thr=iou_thr
    )

    print(f"[OK] Detected {len(top_boxes)} instances after NMS:")
    for i, (b, s, l) in enumerate(zip(top_boxes, top_scores, top_labels)):
        print(f"  Instance {i + 1}: label={text_queries[l]} score={s:.3f} box={b.astype(int).tolist()}")

    # 5) 生成改进的heatmap
    print(f"[INFO] Generating heatmap using method: {heatmap_method}")
    if heatmap_method == 'v2':
        heatmap = boxes_to_heatmap_v2((H, W), top_boxes, top_scores, sigma_ratio=0.15)
    elif heatmap_method == 'edge_based':
        heatmap = boxes_to_heatmap_edge_based(image_np, top_boxes, top_scores,
                                              blur_size=71, edge_weight=2.5)
    else:
        raise ValueError(f"Unknown heatmap_method: {heatmap_method}")

    # 增强对比度
    heatmap = enhance_heatmap(heatmap, clip_percentile=60, gamma=2.5)

    # 保存纯heatmap(灰度图)
    cv2.imwrite("owlv2_box_heatmap.png", (heatmap * 255).astype(np.uint8))
    print("[OK] Saved grayscale heatmap to: owlv2_box_heatmap.png")

    # 6) 从heatmap提取关键点
    points = points_from_heatmap_and_boxes(heatmap, top_boxes)
    print(f"[OK] Extracted {len(points)} keypoints:", points.tolist())

    # 7) SAM分割
    print("[INFO] Running SAM segmentation...")
    masks, sam_scores = sam_masks_from_points(sam_predictor, image_np, points)
    print(f"[OK] SAM segmentation done. Scores: {sam_scores.tolist()}")

    # 8) 保存每个实例mask
    for i, m in enumerate(masks):
        cv2.imwrite(f"mask_{i}.png", (m.astype(np.uint8) * 255))
    print(f"[OK] Saved {len(masks)} individual masks: mask_0.png, mask_1.png, ...")

    # 9) 可视化
    print("[INFO] Generating visualizations...")

    # 三合一对比图
    visualize_heatmap_points_enhanced(
        image_np, heatmap, points,
        save_path="owlv2_heatmap_comparison.png",
        alpha=0.7, cmap='jet'  # 使用jet配色方案
    )

    # 论文专用单图
    visualize_for_paper(
        image_np, heatmap, points,
        save_path="heatmap_for_paper.png",
        alpha=0.65, cmap='jet', dpi=300, show_colorbar=False
    )

    # 多mask叠加图
    visualize_multimask(
        image_np, masks, points=None,
        save_path="owlv2_sam_multimask.png"
    )

    print("\n" + "=" * 60)
    print("All done! Generated files:")
    print("  - owlv2_box_heatmap.png (grayscale heatmap)")
    print("  - owlv2_heatmap_comparison.png (3-panel comparison)")
    print("  - heatmap_for_paper.png (paper-ready figure)")
    print("  - owlv2_sam_multimask.png (segmentation overlay)")
    print("  - mask_0.png, mask_1.png, ... (individual masks)")
    print("=" * 60)


if __name__ == "__main__":
    main()
