import argparse, yaml
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import nms
except Exception as e:
        raise RuntimeError("需要 torchvision.ops.nms，请安装与 PyTorch 匹配版本的 torchvision。") from e

# ===== SAU / LMRA（与你的增强脚本一致） =====
class LayerNormSpatial(nn.Module):
    def __init__(self, c, eps=1e-6, affine=True):
        super().__init__(); self.eps=eps; self.affine=affine
        if affine:
            self.weight=nn.Parameter(torch.ones(1,c,1,1))
            self.bias  =nn.Parameter(torch.zeros(1,c,1,1))
    def forward(self,x):
        u=x.mean(dim=(2,3),keepdim=True); v=((x-u)**2).mean(dim=(2,3),keepdim=True)
        xh=(x-u)/torch.sqrt(v+self.eps)
        return xh*self.weight+self.bias if self.affine else xh

class SAU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3=nn.Conv2d(1,1,3,padding=1,bias=True)
        self.conv1=nn.Conv2d(1,1,1,bias=True)
        self.ln=LayerNormSpatial(1,True)
    def forward(self,M):
        alpha=torch.sigmoid(self.conv3(M))
        alpha=self.conv1(alpha)
        alpha=self.ln(alpha)
        alpha=F.relu(alpha)
        alpha=F.avg_pool2d(alpha,5,1,2)  # 去热点
        return alpha

class LMRA(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.gamma_conv=nn.Conv2d(1,1,1,bias=True)
        self.gamma_ln  =LayerNormSpatial(1,True)
        self.eps_conv  =nn.Conv2d(ch,ch,1,bias=False)
        self.eps_bn    =nn.BatchNorm2d(ch,affine=True)
    @staticmethod
    def luma(x_rgb: torch.Tensor) -> torch.Tensor:
        r,g,b=x_rgb[:,0:1],x_rgb[:,1:2],x_rgb[:,2:3]
        return 0.299*r + 0.587*g + 0.114*b
    def forward(self, I, alpha_hat,
                gain_clip=0.6, strength=1.4, eps_scale=0.02,
                guard_mask=None, max_fg_boost=0.25, max_fg_abs=0.90):
        gamma=torch.sigmoid(self.gamma_ln(self.gamma_conv(alpha_hat)))
        raw=strength * gamma * alpha_hat
        gain=1.0 + gain_clip * torch.tanh(raw / max(gain_clip,1e-6))
        with torch.no_grad():
            Y0=self.luma(I); Y1_pred=Y0*gain
            if guard_mask is None:
                thr=alpha_hat.mean().item()*0.6
                guard_mask=(alpha_hat>thr).float()
            guard_mask=(guard_mask>0.2).float()
            fg_pix=guard_mask.sum()
            if fg_pix>0:
                mean0=(Y0*guard_mask).sum()/(fg_pix+1e-6)
                mean1=(Y1_pred*guard_mask).sum()/(fg_pix+1e-6)
                target=torch.clamp(mean0*(1.0+max_fg_boost), max=max_fg_abs)
                scale=torch.clamp(target/(mean1+1e-6), max=1.0)
            else:
                scale=torch.tensor(1.0, device=I.device)
        gain=1.0 + (gain-1.0)*scale
        F_enh=I*gain
        eps=self.eps_bn(self.eps_conv(I))*eps_scale
        F_out=(F_enh+eps).clamp(0.0,1.0)
        return F_out, gamma, gain, guard_mask

class AWE(nn.Module):
    def __init__(self, hidden=16):
        super().__init__(); self.fc1=nn.Linear(1,hidden); self.fc2=nn.Linear(hidden,1)
        nn.init.normal_(self.fc1.weight,std=0.05); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight,std=0.05); nn.init.zeros_(self.fc2.bias)
    def forward(self,S_hat):
        z=S_hat.mean(dim=(2,3),keepdim=False)
        h=torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))

# ===== I/O 工具 =====
EXTS=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
SUFFIXES=["_mask","-mask","_masks","-masks","_seg","-seg","_sam","-sam","_pred","-pred"]

def load_image(path: Path) -> torch.Tensor:
    img=Image.open(path).convert("RGB"); arr=np.array(img,dtype=np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def load_mask(path: Path, size_hw: Tuple[int,int]) -> torch.Tensor:
    m=Image.open(path).convert("L"); H,W=size_hw; m=m.resize((W,H),resample=Image.BILINEAR)
    arr=np.array(m,dtype=np.float32)/255.0
    return torch.from_numpy(arr)[None,None,...].clamp(0.0,1.0)

def canonicalize_stem(stem:str)->str:
    s=stem.lower()
    for suf in SUFFIXES:
        if s.endswith(suf): return s[:-len(suf)]
    return s

def find_mask_for_image(img_path: Path, mask_dir: Path) -> Optional[Path]:
    stem=img_path.stem
    for ext in EXTS:
        p=mask_dir/f"{stem}{ext}"
        if p.exists(): return p
    for suf in SUFFIXES:
        for ext in EXTS:
            p=mask_dir/f"{stem}{suf}{ext}"
            if p.exists(): return p
    canon=canonicalize_stem(stem)
    for ext in EXTS:
        p=mask_dir/f"{canon}{ext}"
        if p.exists(): return p
        p2=mask_dir/f"{canon}_mask{ext}"
        if p2.exists(): return p2
    return None

def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    return (t.detach().clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)

def draw_boxes(img_np: np.ndarray, boxes: np.ndarray, names: List[str]) -> Image.Image:
    img=Image.fromarray(img_np.copy()); draw=ImageDraw.Draw(img)
    for x1,y1,x2,y2,conf,cls in boxes:
        x1,y1,x2,y2=map(float,[x1,y1,x2,y2]); cls=int(cls); conf=float(conf)
        color=(0,255,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        label=names[cls] if 0<=cls<len(names) else f"id{cls}"
        draw.text((x1+2,y1+2), f"{label} {conf:.2f}", fill=color)
    return img

# ===== 叠加可视化（tint/edge/brighten） =====
def _normalize01(t: torch.Tensor) -> torch.Tensor:
    mn=t.amin(dim=(2,3),keepdim=True); mx=t.amax(dim=(2,3),keepdim=True)
    return (t - mn) / (mx - mn + 1e-6)

def overlay_map_on_image(img_np: np.ndarray, fmap: torch.Tensor,
                         alpha: float=0.35, color: str="r", mode: str="tint") -> np.ndarray:
    """
    mode:
      - 'tint': 按颜色着色透明叠加
      - 'edge': 只画等值边缘线
      - 'brighten': 仅提亮（不着色），out = out * (1 + alpha * fmap)
    """
    m=fmap.detach().clamp(0,1).squeeze().cpu().numpy().astype(np.float32)  # HxW
    out=img_np.astype(np.float32).copy()

    if mode=="edge":
        e=np.zeros_like(m, dtype=bool)
        thr=0.05
        e[:-1,:] |= (np.abs(m[1:,:]-m[:-1,:])>thr)
        e[1: ,:] |= (np.abs(m[1:,:]-m[:-1,:])>thr)
        e[:,:-1] |= (np.abs(m[:,1:]-m[:,:-1])>thr)
        e[:,1: ] |= (np.abs(m[:,1:]-m[:,:-1])>thr)
        col={"r":[255,0,0],"g":[0,255,0],"b":[0,0,255]}[color]
        out[e]=np.array(col, dtype=np.float32)
    elif mode=="brighten":
        factor = 1.0 + float(alpha) * m[..., None]  # HxWx1
        out = out * factor
    else:  # 'tint'
        col={"r":[255,0,0],"g":[0,255,0],"b":[0,0,255]}[color]
        tint=np.stack([m*col[0], m*col[1], m*col[2]], axis=-1)
        out = (1.0-float(alpha))*out + float(alpha)*tint

    return np.clip(out,0,255).astype(np.uint8)

# ===== YOLO 预测 =====
def run_yolo_predict(model, img_tensor: torch.Tensor, imgsz=640, conf=0.25, iou=0.55):
    img_np=to_numpy_img(img_tensor)
    with torch.inference_mode():
        res=model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    r0=res[0]
    if r0.boxes is None or r0.boxes.xyxy.numel()==0:
        return np.zeros((0,6), dtype=np.float32)
    xyxy=r0.boxes.xyxy.cpu().numpy()
    confs=r0.boxes.conf.cpu().numpy()[:,None]
    clss=r0.boxes.cls.cpu().numpy()[:,None]
    return np.concatenate([xyxy,confs,clss],axis=1).astype(np.float32)

# ===== 融合（成对匹配 + 加权） =====
def _logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p=p.clamp(eps, 1.0-eps); return torch.log(p/(1.0-p))
def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return 1.0/(1.0+torch.exp(-x))

def fuse_pairwise_weighted(
    boxes_raw: torch.Tensor, boxes_enh: torch.Tensor, lam: float,
    iou_match: float=0.55, iou_nms: float=0.55, conf_mode: str="avg"
):
    if boxes_raw.numel()==0: return boxes_enh
    if boxes_enh.numel()==0: return boxes_raw
    A,B=boxes_raw, boxes_enh
    out=[]; usedB=torch.zeros((B.shape[0],), dtype=torch.bool, device=B.device)
    for i in range(A.shape[0]):
        a_xyxy=A[i:i+1,:4]; a_conf=A[i,4]; a_cls=A[i,5]
        inter_x1=torch.max(a_xyxy[:,0],B[:,0]); inter_y1=torch.max(a_xyxy[:,1],B[:,1])
        inter_x2=torch.min(a_xyxy[:,2],B[:,2]); inter_y2=torch.min(a_xyxy[:,3],B[:,3])
        inter=(inter_x2-inter_x1).clamp(0)*(inter_y2-inter_y1).clamp(0)
        area_a=(a_xyxy[:,2]-a_xyxy[:,0])*(a_xyxy[:,3]-a_xyxy[:,1])
        area_b=(B[:,2]-B[:,0])*(B[:,3]-B[:,1])
        iou=inter/(area_a+area_b-inter+1e-6)
        j=torch.argmax(iou)
        matched=(iou[j]>=iou_match) and (int(a_cls.item())==int(B[j,5].item())) and (not usedB[j])
        if matched:
            b=B[j]
            box=(1.0-lam)*A[i,:4] + lam*b[:4]
            if conf_mode=="avg":
                conf=(1.0-lam)*a_conf + lam*b[4]
            elif conf_mode=="logit":
                conf=_sigmoid((1.0-lam)*_logit(a_conf) + lam*_logit(b[4]))
            else:
                conf=torch.maximum(a_conf,b[4])
            out.append(torch.stack([box[0],box[1],box[2],box[3],conf.clamp(0,1),a_cls]))
            usedB[j]=True
        else:
            out.append(A[i])
    for k in range(B.shape[0]):
        if not usedB[k]: out.append(B[k])
    allb=torch.stack(out,dim=0) if len(out)>0 else torch.zeros((0,6), device=A.device)
    keep=nms(allb[:,:4], allb[:,4], iou_nms)
    return allb[keep]

# ===== 主程序 =====
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out-normal", type=str, required=True)
    ap.add_argument("--out-enh", type=str, required=True)
    ap.add_argument("--mask-subdir", type=str, default="mask")
    ap.add_argument("--mode", type=str, choices=["enh_only","fuse"], default="fuse")
    ap.add_argument("--awe-weights", type=str, default="")
    ap.add_argument("--lambda-const", type=float, default=None)
    ap.add_argument("--conf-mode", type=str, choices=["avg","logit","max"], default="avg")
    # 叠加可视化
    ap.add_argument("--overlay", type=str, choices=["none","alpha","gamma","gain","guard"], default="none")
    ap.add_argument("--overlay-mode", type=str, choices=["tint","brighten","edge"], default="tint")
    ap.add_argument("--overlay-alpha", type=float, default=0.35)
    ap.add_argument("--overlay-color", type=str, choices=["r","g","b"], default="r")
    ap.add_argument("--overlay-edge", action="store_true", help="兼容旧参数：等价于 --overlay-mode edge")
    # YOLO
    ap.add_argument("--imgs", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.55)
    # SAU/LMRA 参数
    ap.add_argument("--alpha-boost", type=float, default=1.0)
    ap.add_argument("--gain-clip",  type=float, default=0.6)
    ap.add_argument("--strength",   type=float, default=1.4)
    ap.add_argument("--eps-scale",  type=float, default=0.02)
    ap.add_argument("--max-fg-boost", type=float, default=0.25)
    ap.add_argument("--max-fg-abs",   type=float, default=0.90)
    args=ap.parse_args()

    if args.overlay_edge:
        args.overlay_mode="edge"

    data=yaml.safe_load(open(args.data,"r"))
    names=data.get("names",["obj"])
    test_rel=data.get("test")
    if test_rel is None:
        raise SystemExit("[ERR] data.yaml 未定义 test 路径")
    img_dir=(Path(args.data).parent/test_rel).resolve()
    mask_dir=img_dir.parent/args.mask_subdir

    out_normal=Path(args.out_normal); out_enh=Path(args.out_enh)
    out_normal.mkdir(parents=True, exist_ok=True)
    out_enh.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model=YOLO(args.weights)

    sau=SAU().eval()
    lmra=LMRA().eval()
    awe=None
    if args.mode=="fuse":
        awe=AWE().eval()
        if args.lambda_const is None and not args.awe_weights:
            print("[WARN] 未提供 --awe-weights，且未指定 --lambda-const，默认 λ=0.5")
        if args.awe_weights:
            ckpt=torch.load(args.awe_weights, map_location="cpu")
            awe.load_state_dict(ckpt.get("awe", ckpt), strict=False)
            awe.eval()
            print(f"[INFO] Loaded AWE weights: {args.awe_weights}")

    img_paths=[p for p in img_dir.iterdir() if p.suffix.lower() in EXTS]
    print(f"[TEST] images={len(img_paths)}  masks from: {mask_dir}")

    for p in img_paths:
        I=load_image(p); _,_,H,W=I.shape
        mpath=find_mask_for_image(p,mask_dir)
        M=load_mask(mpath,(H,W)) if mpath is not None else None

        # 原图 → out-normal
        pred_raw=run_yolo_predict(model, I, imgsz=args.imgs, conf=args.conf, iou=args.iou)
        vis_raw =draw_boxes(to_numpy_img(I), pred_raw, names)
        vis_raw.save(out_normal/p.name)

        if M is None:
            # 无 mask：增强侧回退为原图
            vis_raw.save(out_enh/p.name)
            continue

        # SAU+LMRA 增强（前景更亮）
        with torch.inference_mode():
            alpha_hat=sau(M)*args.alpha_boost
            I_enh, gamma_map, gain_map, guard_mask = lmra(
                I, alpha_hat,
                gain_clip=args.gain_clip, strength=args.strength, eps_scale=args.eps_scale,
                guard_mask=(M>0.2).float(),
                max_fg_boost=args.max_fg_boost, max_fg_abs=args.max_fg_abs
            )

        # 在增强图上跑 YOLO
        pred_enh=run_yolo_predict(model, I_enh, imgsz=args.imgs, conf=args.conf, iou=args.iou)

        # AWE 融合（可选），仍画在“增强图”上
        if args.mode=="fuse":
            if args.lambda_const is not None:
                lam=float(np.clip(args.lambda_const,0,1))
            elif args.awe_weights:
                with torch.no_grad():
                    lam=float(awe(alpha_hat).item())
            else:
                lam=0.5
            device=torch.device("cpu")
            t_raw=torch.from_numpy(pred_raw).to(device) if pred_raw.size>0 else torch.zeros((0,6), device=device)
            t_enh=torch.from_numpy(pred_enh).to(device) if pred_enh.size>0 else torch.zeros((0,6), device=device)
            fused=fuse_pairwise_weighted(t_raw, t_enh, lam=lam,
                                         iou_match=args.iou, iou_nms=args.iou, conf_mode=args.conf_mode)
            boxes_np=fused.detach().cpu().numpy()
        else:
            boxes_np=pred_enh

        # 叠加（可选）：默认 none；如需“只提亮”选择 --overlay-mode brighten
        enh_np=to_numpy_img(I_enh)
        if args.overlay!="none":
            fmap={"alpha":alpha_hat, "gamma":gamma_map, "gain":gain_map, "guard":guard_mask}[args.overlay]
            fmap=_normalize01(fmap)
            enh_np=overlay_map_on_image(enh_np, fmap, alpha=args.overlay_alpha,
                                        color=args.overlay_color, mode=args.overlay_mode)

        vis_enh=draw_boxes(enh_np, boxes_np, names)
        vis_enh.save(out_enh/p.name)

    print("\nDone.")

if __name__=="__main__":
    main()
