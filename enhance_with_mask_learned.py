import numpy as np
from pathlib import Path
from typing import Tuple
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================== 基础模块（与训练版同构） ===================
class LayerNormSpatial(nn.Module):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        u = x.mean(dim=(2,3), keepdim=True)
        v = ((x - u) ** 2).mean(dim=(2,3), keepdim=True)
        xhat = (x - u) / torch.sqrt(v + self.eps)
        if self.affine:
            xhat = xhat * self.weight + self.bias
        return xhat

class SAU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3 = nn.Conv2d(1, 1, 3, padding=1, bias=True)
        self.conv1 = nn.Conv2d(1, 1, 1, bias=True)
        self.ln    = LayerNormSpatial(1, affine=True)
    def forward(self, M):
        # 轻度平滑+归一，避免mask锯齿导致局部爆亮
        alpha = torch.sigmoid(self.conv3(M))
        alpha = self.conv1(alpha)
        alpha = self.ln(alpha)
        alpha = F.relu(alpha)
        # 再做一次 5x5 平滑以去热点
        alpha = F.avg_pool2d(alpha, kernel_size=5, stride=1, padding=2)
        return alpha   # \hat{alpha}

class LMRA(nn.Module):
    def __init__(self, rgb_channels=3):
        super().__init__()
        self.gamma_conv = nn.Conv2d(1, 1, 1, bias=True)
        self.gamma_ln   = LayerNormSpatial(1, affine=True)
        self.eps_conv   = nn.Conv2d(rgb_channels, rgb_channels, 1, bias=False)
        self.eps_bn     = nn.BatchNorm2d(rgb_channels, affine=True)

    @staticmethod
    def luma(x_rgb: torch.Tensor) -> torch.Tensor:
        # x_rgb: [B,3,H,W]  -> [B,1,H,W], ITU-R BT.601
        r, g, b = x_rgb[:,0:1], x_rgb[:,1:2], x_rgb[:,2:3]
        return 0.299*r + 0.587*g + 0.114*b

    def forward(
        self,
        I, alpha_hat,
        gain_clip=0.6,       # 线性额外增益上限（建议 0.4~1.0）
        strength=1.4,        # 基础力度（建议 1.0~2.0）
        eps_scale=0.02,      # 残差支路强度
        guard_mask=None,     # 用于曝光守卫的前景掩膜（默认用 alpha_hat>阈值）
        max_fg_boost=0.25,   # 前景平均亮度最多提升 25%
        max_fg_abs=0.90      # 前景绝对亮度不能超过 0.90
    ):
        # --- 原始增益（用 tanh 压缩替代硬clamp，防止热点冲顶）---
        gamma = torch.sigmoid(self.gamma_ln(self.gamma_conv(alpha_hat)))   # (0,1)
        raw = strength * gamma * alpha_hat
        gain = 1.0 + gain_clip * torch.tanh(raw / max(gain_clip, 1e-6))    # 平滑上限：1+gain_clip

        # --- 自适应曝光守卫（基于前景平均亮度）---
        with torch.no_grad():
            Y0 = self.luma(I)                                  # 原始亮度
            Y1_pred = Y0 * gain                                # 增强后的亮度预测（不含残差）
            if guard_mask is None:
                # 用 alpha 的动态阈值生成前景区域
                thr = alpha_hat.mean().item() * 0.6
                guard_mask = (alpha_hat > thr).float()
            guard_mask = (guard_mask > 0.2).float()            # [B,1,H,W]
            fg_pix = guard_mask.sum()

            if fg_pix > 0:
                mean0 = (Y0 * guard_mask).sum() / (fg_pix + 1e-6)
                mean1 = (Y1_pred * guard_mask).sum() / (fg_pix + 1e-6)
                # 目标平均亮度：不超过原来的 (1+max_fg_boost)，且绝对值不超过 max_fg_abs
                target = torch.clamp(mean0 * (1.0 + max_fg_boost), max=max_fg_abs)
                scale = torch.clamp(target / (mean1 + 1e-6), max=1.0)      # 只允许降低
            else:
                scale = torch.tensor(1.0, device=I.device)

        gain = 1.0 + (gain - 1.0) * scale                       # 缩放增益避免过曝
        F_enh = I * gain

        # 轻残差稳定
        eps = self.eps_bn(self.eps_conv(I)) * eps_scale
        F_dgp = (F_enh + eps).clamp(0.0, 1.0)
        return F_dgp, gamma, gain, guard_mask

class DGP(nn.Module):
    def __init__(self, rgb_channels=3):
        super().__init__()
        self.sau  = SAU()
        self.lmra = LMRA(rgb_channels=rgb_channels)
    def forward(self, I, M, **kwargs):
        alpha_hat = self.sau(M)
        F_dgp, gamma, gain, gmask = self.lmra(I, alpha_hat, **kwargs)
        return F_dgp, alpha_hat, gamma, gain, gmask

# =================== 兜底算子（无需权重也能增强） ===================
def conv3x3_mean(x: torch.Tensor) -> torch.Tensor:
    k = torch.ones((1, 1, 3, 3), dtype=x.dtype, device=x.device) / 9.0
    return F.conv2d(x, k, padding=1)
def layernorm_spatial_noparam(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = x.mean(dim=(2,3), keepdim=True)
    v = ((x - u) ** 2).mean(dim=(2,3), keepdim=True)
    return (x - u) / torch.sqrt(v + eps)
def fixed_alpha_from_M(M: torch.Tensor) -> torch.Tensor:
    alpha = torch.sigmoid(conv3x3_mean(M))
    alpha = layernorm_spatial_noparam(alpha)
    alpha = F.relu(alpha)
    alpha = F.avg_pool2d(alpha, kernel_size=5, stride=1, padding=2)  # 再平滑一次
    return alpha
def normalize01(x: torch.Tensor, eps=1e-6) -> torch.Tensor:
    mn = x.amin(dim=(2,3), keepdim=True); mx = x.amax(dim=(2,3), keepdim=True)
    return (x - mn) / (mx - mn + eps)

# =================== I/O & 实用 ===================
def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
def load_mask(path: Path, size_hw: Tuple[int, int]) -> torch.Tensor:
    m = Image.open(path).convert("L")
    H, W = size_hw
    m = m.resize((W, H), resample=Image.BILINEAR)
    arr = np.array(m, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None, ...].clamp(0.0, 1.0)
def save_image(t: torch.Tensor, path: Path):
    t = t.clamp(0.0, 1.0)
    arr = (t.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
def save_gray(t: torch.Tensor, path: Path):
    if t.dim() == 4:
        t = t[:, 0:1].squeeze(0)
    t = t.float(); t = t / (t.max() + 1e-6)
    arr = (t.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

SUFFIXES = ["_mask", "-mask", "_masks", "-masks", "_seg", "-seg", "_sam", "-sam", "_pred", "-pred"]
def canonicalize_stem(stem: str) -> str:
    s = stem.lower()
    for suf in SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s
def build_mask_map(mask_dir: Path, extensions) -> dict:
    mask_map = {}
    for m in mask_dir.iterdir():
        if m.is_file() and m.suffix.lower() in extensions:
            raw = m.stem; canon = canonicalize_stem(raw)
            mask_map[raw.lower()] = m; mask_map[canon] = m
    return mask_map

# =================== 主流程 ===================
@torch.no_grad()
def main(
    img_dir="/home/xuhq/project/Afanmingzhuanli/Raccoon.v2-raw.yolov11/valid/images/",
    mask_dir="/home/xuhq/project/Afanmingzhuanli/Raccoon.v2-raw.yolov11/valid/mask/",
    out_dir="/home/xuhq/project/Afanmingzhuanli/Raccoon.v2-raw.yolov11/valid/newimages/",
    dgp_ckpt="/home/xuhq/project/Afanmingzhuanli/dgp_saulmra.pth",
    mode="fixed",          # 'auto' / 'learned' / 'fixed' / 'mask_direct'
    # —— 温和默认参数 —— #
    gain_clip=0.6,
    strength=1.4,
    eps_scale=0.02,
    alpha_boost=1.0,
    max_fg_boost=0.25,
    max_fg_abs=0.90,
    save_vis=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
):
    img_dir = Path(img_dir); mask_dir = Path(mask_dir); out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / "vis";  vis_dir.mkdir(parents=True, exist_ok=True) if save_vis else None

    mask_map = build_mask_map(mask_dir, extensions)

    dgp = None
    if mode in ("learned", "auto"):
        ckpt = torch.load(dgp_ckpt, map_location=device)
        dgp = DGP(rgb_channels=ckpt.get("meta", {}).get("rgb_channels", 3)).to(device).eval()
        dgp.sau.load_state_dict(ckpt["sau"], strict=False)
        dgp.lmra.load_state_dict(ckpt["lmra"], strict=False)

    total, used = 0, 0
    for p in img_dir.iterdir():
        if not (p.is_file() and p.suffix.lower() in extensions):
            continue
        total += 1

        stem_raw = p.stem
        stem_can = canonicalize_stem(stem_raw)
        mpath = mask_map.get(stem_raw.lower()) or mask_map.get(stem_can)
        if mpath is None:
            print(f"[WARN] mask not found for image: {p.name}")
            continue

        I = load_image(p).to(device)
        _, _, H, W = I.shape
        M = load_mask(mpath, (H, W)).to(device)

        # ========== 选择 α̂ 来源 ==========
        if mode == "mask_direct":
            alpha_hat = normalize01(M) * alpha_boost
            gamma_map = torch.ones_like(alpha_hat) * 0.9
        elif mode == "fixed":
            alpha_hat = fixed_alpha_from_M(M) * alpha_boost
            gamma_map = torch.sigmoid(layernorm_spatial_noparam(alpha_hat))
        else:
            F_tmp, alpha_hat, gamma_map, _, _ = dgp(I, M, gain_clip=gain_clip,
                                                    strength=strength, eps_scale=eps_scale,
                                                    max_fg_boost=max_fg_boost, max_fg_abs=max_fg_abs)
            if mode == "auto" and alpha_hat.abs().max().item() < 1e-4:
                print(f"[AUTO] {p.name}: learned alpha≈0 → fallback to fixed")
                alpha_hat = fixed_alpha_from_M(M)
                gamma_map = torch.sigmoid(layernorm_spatial_noparam(alpha_hat))
            alpha_hat = alpha_hat * alpha_boost

        # ========== 用带守卫的 LMRA 合成 ==========
        lmra_only = LMRA(rgb_channels=3).to(device).eval()
        F_dgp, _, gain_map, gmask = lmra_only(
            I, alpha_hat,
            gain_clip=gain_clip, strength=strength, eps_scale=eps_scale,
            guard_mask=(M>0.2).float(),    # 用原始mask做守卫区域
            max_fg_boost=max_fg_boost, max_fg_abs=max_fg_abs
        )

        # 保存
        save_path = out_dir / p.name
        save_image(F_dgp, save_path)

        if save_vis:
            def save_gray_auto(t, path): save_gray(t, path)
            save_gray_auto(alpha_hat, vis_dir / f"{p.stem}_alpha.png")
            save_gray_auto(gamma_map, vis_dir / f"{p.stem}_gamma.png")
            diff = (F_dgp - I).abs().mean(dim=1, keepdim=True)
            save_gray_auto(diff,      vis_dir / f"{p.stem}_diff.png")
            save_gray_auto(gain_map,  vis_dir / f"{p.stem}_gain.png")
            save_gray_auto(gmask,     vis_dir / f"{p.stem}_guardmask.png")

        used += 1
        print(f"[OK] {p.name} -> {save_path}")

    print(f"\nDone. processed={total}, enhanced={used}, skipped(without mask)={total-used}")

if __name__ == "__main__":
    main()
