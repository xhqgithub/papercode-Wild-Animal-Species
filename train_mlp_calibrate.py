import argparse, yaml, os, json, math
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms

# ====== MSA / FER / AWE（与推理版完全对齐；FER 含防过曝守卫）======
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

class MSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c3=nn.Conv2d(1,1,3,padding=1,bias=True)
        self.c1=nn.Conv2d(1,1,1,bias=True)
        self.ln=LayerNormSpatial(1,True)
    def forward(self,M, alpha_boost:float=1.0):
        s=torch.sigmoid(self.c3(M))
        s=self.c1(s); s=self.ln(s); s=F.relu(s)
        s=F.avg_pool2d(s,5,1,2)
        if alpha_boost!=1.0: s = s * alpha_boost
        return s

class FER(nn.Module):
    def __init__(self, ch=3):
        super().__init__()
        self.d_c1=nn.Conv2d(1,1,1,bias=True)
        self.d_ln=LayerNormSpatial(1,True)
        self.r_c1=nn.Conv2d(ch,ch,1,bias=False)
        self.r_bn=nn.BatchNorm2d(ch,affine=True)

    @staticmethod
    def luma(x_rgb: torch.Tensor) -> torch.Tensor:
        r,g,b=x_rgb[:,0:1],x_rgb[:,1:2],x_rgb[:,2:3]
        return 0.299*r + 0.587*g + 0.114*b

    def forward(self,I,S_hat,
                gain_clip=0.6, strength=1.4, eps_scale=0.02,
                guard_mask=None, max_fg_boost=0.25, max_fg_abs=0.90):
        delta = torch.sigmoid(self.d_ln(self.d_c1(S_hat)))
        raw   = strength * delta * S_hat
        gain  = 1.0 + gain_clip * torch.tanh(raw / max(gain_clip,1e-6))

        with torch.no_grad():
            Y0 = self.luma(I)
            Y1 = Y0 * gain
            if guard_mask is None:
                thr = S_hat.mean().item() * 0.6
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
        eps   = self.r_bn(self.r_c1(I)) * eps_scale
        return (F_mod + eps).clamp(0.0, 1.0)

class AWE(nn.Module):
    def __init__(self, hidden=16):
        super().__init__(); self.fc1=nn.Linear(1,hidden); self.fc2=nn.Linear(hidden,1)
        nn.init.normal_(self.fc1.weight,std=0.05); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight,std=0.05); nn.init.zeros_(self.fc2.bias)
    def forward(self,S_hat):
        z=S_hat.mean(dim=(2,3),keepdim=False)  # [B,1]
        h=F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h))      # [B,1] in (0,1)

# ====== I/O 与工具 ======
EXTS=(".jpg",".jpeg",".png",".bmp",".tif",".tiff")
SUFFIXES=["_mask","-mask","_masks","-masks","_seg","-seg","_sam","-sam","_pred","-pred"]

def load_image(p:Path)->torch.Tensor:
    img=Image.open(p).convert("RGB"); arr=np.array(img,dtype=np.float32)/255.0
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)

def load_mask(p:Path,size_hw:Tuple[int,int])->torch.Tensor:
    m=Image.open(p).convert("L"); H,W=size_hw; m=m.resize((W,H),resample=Image.BILINEAR)
    arr=np.array(m,dtype=np.float32)/255.0
    return torch.from_numpy(arr)[None,None,...].clamp(0,1)

def canonicalize_stem(stem:str)->str:
    s=stem.lower()
    for suf in SUFFIXES:
        if s.endswith(suf): return s[:-len(suf)]
    return s

def find_mask_for_image(img_path:Path, mask_dir:Path)->Optional[Path]:
    stem=img_path.stem
    for ext in EXTS:
        p=mask_dir/f"{stem}{ext}";     p2=mask_dir/f"{stem}_mask{ext}"
        if p.exists(): return p
        if p2.exists(): return p2
    canon=canonicalize_stem(stem)
    for ext in EXTS:
        p=mask_dir/f"{canon}{ext}"; p2=mask_dir/f"{canon}_mask{ext}"
        if p.exists(): return p
        if p2.exists(): return p2
    return None

def to_numpy_img(t: torch.Tensor) -> np.ndarray:
    t = (t.detach().clamp(0,1).squeeze(0).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return t

def run_yolo_predict(model, img_tensor:torch.Tensor, imgsz=640, conf=0.25, iou=0.55, device="cuda"):
    t = to_numpy_img(img_tensor)
    with torch.inference_mode():
        res = model.predict(t, imgsz=imgsz, conf=conf, iou=iou, verbose=False, device=device)
    r = res[0]
    if r.boxes is None or r.boxes.xyxy.numel()==0:
        return np.zeros((0,6),np.float32)
    xyxy = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()[:,None]
    clss  = r.boxes.cls.cpu().numpy()[:,None]
    return np.concatenate([xyxy,confs,clss],axis=1).astype(np.float32)

def fuse_simple(boxes_raw:torch.Tensor, boxes_enh:torch.Tensor, lam:float, conf_thres=0.25, iou_thres=0.55):
    if boxes_raw.numel()==0 and boxes_enh.numel()==0:
        return torch.zeros((0,6))
    if boxes_raw.numel()>0: boxes_raw[:,4]*=(1.0-lam)
    if boxes_enh.numel()>0: boxes_enh[:,4]*=lam
    allb=torch.cat([b for b in [boxes_raw,boxes_enh] if b.numel()>0], dim=0)
    keep=nms(allb[:,:4], allb[:,4], iou_thres)
    fused=allb[keep]
    return fused[fused[:,4] >= conf_thres] if fused.numel()>0 else fused

def load_gt_yolo(txt:Path)->np.ndarray:
    if not txt.exists(): return np.zeros((0,6),np.float32)
    rows=[]
    for line in txt.read_text().splitlines():
        if not line.strip(): continue
        parts=line.strip().split()
        if len(parts)<5: continue
        c,cx,cy,w,h = map(float,parts[:5])
        rows.append([c,cx,cy,w,h])
    if not rows: return np.zeros((0,6),np.float32)
    return np.array(rows,np.float32)

def xywhn_to_xyxy(arr, W, H):
    if arr.size==0: return np.zeros((0,6),np.float32)
    c=arr[:,0]; cx=arr[:,1]*W; cy=arr[:,2]*H; w=arr[:,3]*W; h=arr[:,4]*H
    x1=cx-w/2; y1=cy-h/2; x2=cx+w/2; y2=cy+h/2
    out=np.stack([x1,y1,x2,y2,np.ones_like(c),c],axis=1).astype(np.float32)
    return out

def iou_matrix(A:np.ndarray,B:np.ndarray)->np.ndarray:
    if A.size==0 or B.size==0: return np.zeros((A.shape[0], B.shape[0]), np.float32)
    x11,y11,x12,y12=A[:,0],A[:,1],A[:,2],A[:,3]
    x21,y21,x22,y22=B[:,0],B[:,1],B[:,2],B[:,3]
    inter_x1=np.maximum(x11[:,None],x21[None,:]); inter_y1=np.maximum(y11[:,None],y21[None,:])
    inter_x2=np.minimum(x12[:,None],x22[None,:]); inter_y2=np.minimum(y12[:,None],y22[None,:])
    inter=np.clip(inter_x2-inter_x1,0,None)*np.clip(inter_y2-inter_y1,0,None)
    area1=(x12-x11)*(y12-y11); area2=(x22-x21)*(y22-y21)
    return inter/(area1[:,None]+area2[None,:]-inter+1e-6)

def score_predictions(pred:np.ndarray, gt:np.ndarray, iou_thr=0.5, fp_penalty=0.2)->float:
    if pred.size==0 and gt.size==0: return 0.0
    if pred.size==0: return -0.5*len(gt)
    if gt.size==0:   return -fp_penalty*pred.shape[0]
    score=0.0
    for c in np.unique(pred[:,5]).astype(int):
        P=pred[pred[:,5]==c]; G=gt[gt[:,5]==c]
        if G.size==0:
            score -= fp_penalty*P.shape[0]
            continue
        iou=iou_matrix(P[:,:4], G[:,:4])
        used_g=np.zeros((G.shape[0],),bool)
        order=np.argsort(-P[:,4])
        fp=0; tp_sum=0.0
        for i in order:
            j=np.argmax(iou[i])
            if iou[i,j]>=iou_thr and (not used_g[j]):
                used_g[j]=True; tp_sum+=P[i,4]
            else:
                fp+=1
        score += tp_sum - fp_penalty*fp
    return float(score)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--split", type=str, default="val", choices=["train","val","test"])
    ap.add_argument("--imgs", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou",  type=float, default=0.55)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=str, default="runs/awe_calib")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam-step", type=float, default=0.05)  # 更细的 λ 网格
    ap.add_argument("--tau", type=float, default=0.5, help="soft target temperature; <=0 使用硬目标")
    ap.add_argument("--mask-subdir", type=str, default="labels/../mask")  # 兼容你的目录结构
    ap.add_argument("--alpha-boost", type=float, default=1.0)
    ap.add_argument("--gain-clip",  type=float, default=0.6)
    ap.add_argument("--strength",   type=float, default=1.4)
    ap.add_argument("--eps-scale",  type=float, default=0.02)
    ap.add_argument("--max-fg-boost", type=float, default=0.25)
    ap.add_argument("--max-fg-abs",   type=float, default=0.90)
    args=ap.parse_args()

    # 解析路径
    data=yaml.safe_load(open(args.data,"r"))
    rel=data[args.split]
    root=Path(args.data).parent
    img_dir   = root / rel
    if "images" in rel:
        label_dir = root / rel.replace("images","labels")
        mask_dir  = root / rel.replace("images","mask")
    else:
        label_dir = img_dir.parent / "labels"
        mask_dir  = img_dir.parent / "mask"

    from ultralytics import YOLO
    model=YOLO(args.weights)

    device=args.device
    msa=MSA().to(device).eval()
    fer=FER().to(device).eval()
    for m in (msa, fer):
        for p in m.parameters():
            p.requires_grad=False

    awe=AWE(hidden=16).to(device).train()
    opt=torch.optim.Adam(awe.parameters(), lr=args.lr)

    img_paths=[p for p in img_dir.iterdir() if p.suffix.lower() in EXTS]
    lams=np.arange(0.0,1.0+1e-9,args.lam_step)

    out_dir=Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        total_loss=0.0; cnt=0
        hist_bins=np.zeros_like(lams, dtype=np.int64)

        for p in img_paths:
            I=load_image(p).to(device); H,W=I.shape[-2:]
            mpath=find_mask_for_image(p,mask_dir)
            if mpath is None:
                continue
            M=load_mask(mpath,(H,W)).to(device)

            with torch.inference_mode():
                S_hat=msa(M, alpha_boost=args.alpha_boost)
                I_enh=fer(I,S_hat,
                          gain_clip=args.gain_clip, strength=args.strength, eps_scale=args.eps_scale,
                          guard_mask=(M>0.2).float(),
                          max_fg_boost=args.max_fg_boost, max_fg_abs=args.max_fg_abs)

            pred_raw = run_yolo_predict(model, I,     imgsz=args.imgs, conf=args.conf, iou=args.iou, device=device)
            pred_enh = run_yolo_predict(model, I_enh, imgsz=args.imgs, conf=args.conf, iou=args.iou, device=device)

            t_raw=torch.from_numpy(pred_raw).to("cpu") if pred_raw.size>0 else torch.zeros((0,6))
            t_enh=torch.from_numpy(pred_enh).to("cpu") if pred_enh.size>0 else torch.zeros((0,6))

            gt_n=load_gt_yolo(label_dir/f"{p.stem}.txt")
            gt_xyxy=xywhn_to_xyxy(gt_n,W,H)

            # 网格搜索 λ → 评分
            scores=[]
            for lam in lams:
                fused=fuse_simple(t_raw.clone(), t_enh.clone(), lam=float(lam),
                                  conf_thres=args.conf, iou_thres=args.iou)
                fused_np=fused.detach().cpu().numpy()
                sc=score_predictions(fused_np, gt_xyxy, iou_thr=0.5, fp_penalty=0.2)
                scores.append(sc)
            scores=np.array(scores, dtype=np.float32)
            j=int(np.argmax(scores))
            lam_best=float(lams[j]); hist_bins[j]+=1

            # 目标 λ：硬目标或软目标（softmax 期望）
            if args.tau is not None and args.tau>0:
                w = np.exp((scores - scores.max())/max(args.tau,1e-6))
                w = w / (w.sum() + 1e-9)
                lam_t = float((w * lams).sum())
            else:
                lam_t = lam_best

            lam_pred=awe(S_hat)  # [1,1]
            target=torch.tensor([[lam_t]], dtype=torch.float32, device=lam_pred.device)
            loss=F.mse_loss(lam_pred, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss+=loss.item(); cnt+=1

        # 直方图 + 日志
        hist_line=" ".join([f"{l:.2f}:{int(n)}" for l,n in zip(lams, hist_bins)])
        print(f"Epoch {ep+1}/{args.epochs}  loss={total_loss/max(cnt,1):.6f}  | λ-hist [{hist_line}]")

    # 保存权重 + 元信息
    meta={
        "split": args.split,
        "imgs": args.imgs, "conf": args.conf, "iou": args.iou,
        "lam_step": args.lam_step, "tau": args.tau,
        "alpha_boost": args.alpha_boost,
        "gain_clip": args.gain_clip, "strength": args.strength, "eps_scale": args.eps_scale,
        "max_fg_boost": args.max_fg_boost, "max_fg_abs": args.max_fg_abs,
    }
    torch.save({"awe":awe.state_dict(), "meta":meta}, out_dir/"awe_weights.pth")
    (out_dir/"awe_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("Saved:", (out_dir/"awe_weights.pth").resolve())
    print("Meta :", (out_dir/"awe_meta.json").resolve())

if __name__=="__main__":
    main()
