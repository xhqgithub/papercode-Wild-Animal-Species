import torch
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor

# 路径配置
sam_checkpoint = "/home/xuhq/project/yolosam/sam_vit_h_4b8939.pth"
image_path = "/home/xuhq/project/yolosam/1752232599680.jpg"
output_mask_path = "/home/xuhq/project/yolosam/mask_color.png"

# 1. 加载图片
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 2. 加载SAM模型
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

# 3. 设置图片
predictor.set_image(image)

input_point = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
input_label = np.array([1])  # 1表示前景

# 5. 推理mask
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

# 6. 生成自定义颜色的mask图片
# 定义颜色（RGB）
bg_color = np.array([229, 224, 235], dtype=np.uint8)  # 背景色 #e5e0eb
fg_color = np.array([189, 114, 57], dtype=np.uint8)   # 前景色 #bd7239

# 创建背景图像
color_mask = np.ones_like(image, dtype=np.uint8) * bg_color

# 覆盖前景区域
mask = masks[0]
color_mask[mask] = fg_color

# 7. 保存彩色mask图片
color_mask_img = Image.fromarray(color_mask)
color_mask_img.save(output_mask_path)
print(f"彩色掩码图片已保存到: {output_mask_path}")
