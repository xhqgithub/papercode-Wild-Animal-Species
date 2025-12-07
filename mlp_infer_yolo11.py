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
    raise RuntimeError("需要 torchvision 的 nms。请安装与 PyTorch 匹配版本的 torchvision。") from e

# ============ 模块：SAU、LMRA（带曝光守卫）、FAFW ============
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
        self.c3=nn.Conv2d(1,1,3,padding=1,bias=True)
        self.c1=nn.Conv2d(1,1,1,bias=True)
        self.ln=LayerNormSpatial(1,True)
    def forward(self,M, alpha_boost=1.0):
        s=torch.sigmoid(self.c3(M))
        s=self.c1(s); s=self.ln(s); s=F.relu(s)
        s=F.avg_pool2d(s,5,1,2)  # 轻度平滑，去热点
        if alpha_boost!=1.0: s = s * alpha_boost
        return s

class LMRA(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.delta_c1=nn.Conv2d(1,1,1,bias=True)
        self.delta_ln=LayerNormSpatial(1,True)
        self.res_c1  =nn.Conv2d(ch,ch,1,bias=False)
        self.res_bn  =nn.BatchNorm2d(ch,affine=True)

    @staticmethod
    def luma(x_rgb: torch.Tensor) -> torch.Tensor:
        r,g,b=x_rgb[:,0:1],x_rgb[:,1:2],x_rgb[:,2:3]
        return 0.299*r + 0.587*g + 0.114*b

    def forward(self, I, S_hat,
                gain_clip=0.6, strength=1.4, eps_scale=0.02,
                guard_mask=None, max_fg_boost=0.25, max_fg_abs=0.90):
        delta = torch.sigmoid(self.delta_ln(self.delta_c1(S_hat)))
        raw   = strength * delta * S_hat
        gain  = 1.0 + gain_clip * torch.tanh(raw / max(gain_clip,1e-6))

        # 曝光守卫，限制前景平均亮度提升与绝对上限
        with torch.no_grad():
            Y0 = self.luma(I)
            Y1 = Y0 * gain
            if guard_mask is None:
                thr = S_hat.mean().item()*0.6
                guard_mask = (S_hat > thr).float()
            guard_mask = (guard_mask > 0.2).float()
            fg_pix = guard_mask.sum()
            if fg_pix > 0:
                mean0 = (Y0 * guard_mask).sum() / (fg_pix + 1e-6)
                mean1 = (Y1 * guard_mask).sum() / (fg_pix + 1e-6)
                target = torch.clamp(mean0 * (1.0 + max_fg_boost), max=max_fg_abs)
                scale  = torch.clamp(target / (mean1 + 1e-6), max=1.0)
            else:
                scale = torch.tensor(1.0, device=I.device)

        gain = 1.0 + (gain - 1.0) * scale
        F_mod = I * gain
        eps   = self.res_bn(self.res_c1(I)) * eps_scale
        F_out = (F_mod + eps).clamp(0.0, 1.0)
        return F_out, delta, gain, guard_mask

class FAFW(nn.Module):
    def __init__(self, hidden=16):
        super().__init__(); self.fc1=nn.Linear(1,hidden); self.fc2=nn.Linear(hidden,1)
        nn.init.normal_(self.fc1.weight,std=0.05); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight,std=0.05); nn.init.zeros_(self.fc2.bias)
    def forward(self,S_hat):
        z=S_hat.mean(dim=(2,3),keepdim=False)  # [B,1]
        h=torch.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))      # [B,1]

# ============ I/O 与工具 ============
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
    # 1) 完全同名
    for ext in EXTS:
        p=mask_dir/f"{stem}{ext}"
        if p.exists(): return p
    # 2) 常见后缀
    for suf in SUFFIXES:
        for ext in EXTS:
            p=mask_dir/f"{stem}{suf}{ext}"
            if p.exists(): return p
    # 3) 规范化 stem
    canon=canonicalize_stem(stem)
    for ext in EXTS:
        p=mask_dir/f"{canon}{ext}"
        if p.exists(): return p
        p2=mask_dir/f"{canon}_mask{ext}"
        if p2.exists(): return p2
    return None

def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    # 修复：detach 后再转 numpy，避免 requires_grad=True 报错
    t = t.detach().clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()
    return (t*255).round().astype(np.uint8)

def run_yolo_predict(model, img_tensor: torch.Tensor, imgsz=640, conf=0.25, iou=0.55):
    # 兜底：detach，防止上游忘了关闭梯度
    img_tensor = img_tensor.detach()
    img_np = to_numpy_img(img_tensor)  # HWC, uint8
    res = model.predict(img_np, imgsz=imgsz, conf=conf, iou=iou, verbose=False)
    r0 = res[0]
    if r0.boxes is None or r0.boxes.xyxy.numel()==0:
        return np.zeros((0,6), dtype=np.float32)
    xyxy = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()[:,None]
    clss  = r0.boxes.cls.cpu().numpy()[:,None]
    return np.concatenate([xyxy,confs,clss],axis=1).astype(np.float32)

# ============ 融合策略 ============
def fuse_simple(boxes_raw: torch.Tensor, boxes_enh: torch.Tensor, lam: float, iou_thres=0.55):
    if boxes_raw.numel()==0 and boxes_enh.numel()==0:
        dev = torch.device("cpu")
        return torch.zeros((0,6), device=dev)
    if boxes_raw.numel()>0: boxes_raw[:,4] *= (1.0 - lam)
    if boxes_enh.numel()>0: boxes_enh[:,4] *= lam
    allb = torch.cat([b for b in (boxes_raw, boxes_enh) if b.numel()>0], dim=0)
    keep = nms(allb[:,:4], allb[:,4], iou_thres)
    return allb[keep]

def fuse_pairwise(boxes_raw: torch.Tensor, boxes_enh: torch.Tensor, lam: float, iou_match=0.55, iou_nms=0.55):
    if boxes_raw.numel()==0: return boxes_enh
    if boxes_enh.numel()==0: return boxes_raw
    A,B=boxes_raw, boxes_enh
    out=[]; usedB=torch.zeros((B.shape[0],),dtype=torch.bool,device=B.device)
    for i in range(A.shape[0]):
        a=A[i:i+1,:4]; ac=A[i,4]; acl=A[i,5]
        inter_x1=torch.max(a[:,0],B[:,0]); inter_y1=torch.max(a[:,1],B[:,1])
        inter_x2=torch.min(a[:,2],B[:,2]); inter_y2=torch.min(a[:,3],B[:,3])
        inter=(inter_x2-inter_x1).clamp(0)*(inter_y2-inter_y1).clamp(0)
        area_a=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); area_b=(B[:,2]-B[:,0])*(B[:,3]-B[:,1])
        iou=inter/(area_a + area_b - inter + 1e-6)
        j=torch.argmax(iou)
        if iou[j]>=iou_match and (int(acl)==int(B[j,5].item())) and not usedB[j]:
            b=B[j]
            box=lam*b[:4] + (1-lam)*A[i,:4]
            conf=lam*b[4] + (1-lam)*ac
            cls=A[i,5]
            out.append(torch.stack([box[0],box[1],box[2],box[3],conf,cls]))
            usedB[j]=True
        else:
            out.append(A[i])
    for k in range(B.shape[0]):
        if not usedB[k]: out.append(B[k])
    allb=torch.stack(out,dim=0) if len(out)>0 else torch.zeros((0,6),device=A.device)
    keep=nms(allb[:,:4],allb[:,4],iou_nms)
    return allb[keep]

# ============ 可视化 ============
def draw_boxes(img_np: np.ndarray, boxes: np.ndarray, names: List[str]) -> Image.Image:
    img=Image.fromarray(img_np.copy()); draw=ImageDraw.Draw(img)
    for x1,y1,x2,y2,conf,cls in boxes:
        x1,y1,x2,y2=map(float,[x1,y1,x2,y2]); cls=int(cls); conf=float(conf)
        color=(0,255,0)
        draw.rectangle([x1,y1,x2,y2], outline=color, width=2)
        label = names[cls] if 0 <= cls < len(names) else f"id{cls}"
        draw.text((x1+2,y1+2), f"{label} {conf:.2f}", fill=color)
    return img

# ============ 主流程 ============
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--imgs", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.55)
    ap.add_argument("--out", type=str, default="runs/FAFW_inLMRA")
    ap.add_argument("--fuse", type=str, choices=["simple","pair"], default="simple")
    ap.add_argument("--FAFW-weights", type=str, default="")
    ap.add_argument("--lambda-const", type=float, default=None)
    ap.add_argument("--split", type=str, default="all", choices=["all","train","val","test"])
    ap.add_argument("--mask-subdir", type=str, default="mask")
    # —— 与增强脚本对齐的参数 —— #
    ap.add_argument("--alpha-boost", type=float, default=1.0)
    ap.add_argument("--gain-clip",  type=float, default=0.6)
    ap.add_argument("--strength",   type=float, default=1.4)
    ap.add_argument("--eps-scale",  type=float, default=0.02)
    ap.add_argument("--max-fg-boost", type=float, default=0.25)
    ap.add_argument("--max-fg-abs",   type=float, default=0.90)
    ap.add_argument("--no-vis", action="store_true", help="不保存可视化图")
    args=ap.parse_args()

    data=yaml.safe_load(open(args.data,"r"))
    names=data.get("names",["obj"])
    splits_all={"train":data.get("train"), "val":data.get("val"), "test":data.get("test")}
    splits = {args.split: splits_all.get(args.split)} if args.split!="all" else splits_all

    from ultralytics import YOLO
    model=YOLO(args.weights)  # 会自动选择可用设备

    SAU=SAU().eval()
    LMRA=LMRA().eval()
    FAFW=FAFW(hidden=16).eval()
    if args.FAFW_weights:
        ckpt=torch.load(args.FAFW_weights, map_location="cpu")
        FAFW.load_state_dict(ckpt.get("FAFW", ckpt), strict=False); FAFW.eval()
        print(f"[INFO] Loaded FAFW weights: {args.FAFW_weights}")

    for split, rel in splits.items():
        if not rel: continue
        base_dir=(Path(args.data).parent/rel).resolve()
        img_dir=base_dir
        mask_dir=img_dir.parent/args.mask_subdir
        out_dir=Path(args.out)/split
        (out_dir/"images").mkdir(parents=True, exist_ok=True)
        (out_dir/"labels").mkdir(parents=True, exist_ok=True)

        img_paths=[p for p in img_dir.iterdir() if p.suffix.lower() in EXTS]
        print(f"[{split}] images={len(img_paths)}  masks from: {mask_dir}")

        for p in img_paths:
            I=load_image(p)           # [1,3,H,W]
            _,_,H,W=I.shape
            mpath=find_mask_for_image(p,mask_dir)
            if mpath is None:
                print(f"[WARN] mask not found for {p.name}; skip"); continue
            M=load_mask(mpath,(H,W))  # [1,1,H,W]

            # 关键修复：推理阶段关闭梯度，避免下游 numpy() 报错
            with torch.inLMRAence_mode():
                S_hat=SAU(M, alpha_boost=args.alpha_boost)
                I_enh,_,_,_ = LMRA(I, S_hat,
                                  gain_clip=args.gain_clip, strength=args.strength, eps_scale=args.eps_scale,
                                  guard_mask=(M>0.2).float(),
                                  max_fg_boost=args.max_fg_boost, max_fg_abs=args.max_fg_abs)

            # FAFW λ
            lam = float(np.clip(args.lambda_const,0,1)) if args.lambda_const is not None else float(FAFW(S_hat).item())

            # YOLO 两路预测
            try:
                pred_raw=run_yolo_predict(model,I,    imgsz=args.imgs,conf=args.conf,iou=args.iou)
                pred_enh=run_yolo_predict(model,I_enh,imgsz=args.imgs,conf=args.conf,iou=args.iou)
            except Exception as e:
                print(f"[ERR] YOLO predict failed on {p.name}: {e}")
                pred_raw=np.zeros((0,6),dtype=np.float32)
                pred_enh=np.zeros((0,6),dtype=np.float32)

            device=torch.device("cpu")
            t_raw=torch.from_numpy(pred_raw).to(device) if pred_raw.size>0 else torch.zeros((0,6),device=device)
            t_enh=torch.from_numpy(pred_enh).to(device) if pred_enh.size>0 else torch.zeros((0,6),device=device)

            # 融合
            fused = fuse_simple(t_raw.clone(), t_enh.clone(), lam=lam, iou_thres=args.iou) \
                    if args.fuse=="simple" else \
                    fuse_pairwise(t_raw.clone(), t_enh.clone(), lam=lam, iou_match=args.iou, iou_nms=args.iou)

            fused_np=fused.detach().cpu().numpy()
            if not args.no_vis:
                vis_img=draw_boxes(to_numpy_img(I), fused_np, names)
                vis_img.save(out_dir/"images"/p.name)

            # 保存 YOLO 预测 txt（cls cx cy w h conf）
            h,w=H,W
            def xyxy2yolo(x1,y1,x2,y2):
                cx=(x1+x2)/2.0/w; cy=(y1+y2)/2.0/h; bw=(x2-x1)/w; bh=(y2-y1)/h
                return cx,cy,bw,bh
            lines=[]
            for x1,y1,x2,y2,conf,cls in fused_np:
                cx,cy,bw,bh=xyxy2yolo(x1,y1,x2,y2)
                lines.append(f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(conf):.6f}")
            (out_dir/"labels"/f"{p.stem}.txt").write_text("\n".join(lines))
            print(f"[OK {split}] {p.name}  λ={lam:.3f}  raw={len(pred_raw)} enh={len(pred_enh)} -> fused={len(fused_np)}")

    print("\nDone. Results saved to:", Path(args.out).resolve())

if __name__=="__main__":
    main()
