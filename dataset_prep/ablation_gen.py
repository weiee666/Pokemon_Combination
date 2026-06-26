# -*- coding: utf-8 -*-
"""Ablation comparison: self-attn (exp02 vs exp03) and cross-attn (exp06 vs exp08).
Same names + same initial noise so without/with are directly comparable.
Saves 4 labeled grids to /root/autodl-tmp/ablation/."""
import os, json, math, glob, torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dev = "cuda" if torch.cuda.is_available() else "cpu"
E = "/root/autodl-tmp/experiments"
OUT = "/root/autodl-tmp/ablation"; os.makedirs(OUT, exist_ok=True)
PICK = ["pikachu", "charizard", "bulbasaur", "squirtle", "gengar", "eevee", "snorlax", "mewtwo"]
GUID = 4.0; T = 1000

def sinu(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
    a = t[:, None].float() * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)

class ResBlock(nn.Module):
    def __init__(s, i, o, c, g=8):
        super().__init__()
        s.block1 = nn.Sequential(nn.GroupNorm(g, i), nn.SiLU(), nn.Conv2d(i, o, 3, padding=1))
        s.cond_mlp = nn.Sequential(nn.SiLU(), nn.Linear(c, o))
        s.block2 = nn.Sequential(nn.GroupNorm(g, o), nn.SiLU(), nn.Conv2d(o, o, 3, padding=1))
        s.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
    def forward(s, x, c):
        h = s.block1(x); h = h + s.cond_mlp(c)[:, :, None, None]; h = s.block2(h)
        return h + s.skip(x)

class Attn(nn.Module):
    def __init__(s, ch, h=4):
        super().__init__(); s.norm = nn.GroupNorm(8, ch); s.mha = nn.MultiheadAttention(ch, h, batch_first=True)
    def forward(s, x):
        B, C, H, W = x.shape
        h = s.norm(x).reshape(B, C, H*W).transpose(1, 2); h, _ = s.mha(h, h, h)
        return x + h.transpose(1, 2).reshape(B, C, H, W)

class CrossAttn(nn.Module):
    def __init__(s, ch, ctx=512, h=4):
        super().__init__(); s.norm = nn.GroupNorm(8, ch); s.kv = nn.Linear(ctx, ch); s.mha = nn.MultiheadAttention(ch, h, batch_first=True)
    def forward(s, x, ctx, mask=None):
        B, C, H, W = x.shape
        q = s.norm(x).reshape(B, C, H*W).transpose(1, 2); kv = s.kv(ctx)
        h, _ = s.mha(q, kv, kv, key_padding_mask=mask)
        return x + h.transpose(1, 2).reshape(B, C, H, W)

# ---------- exp02/03: flat base64 label UNet ----------
class UNetFlat(nn.Module):
    def __init__(s, base=64, c=256, num_classes=20, attn=False):
        super().__init__(); s.c_dim = c; s.num_classes = num_classes; s.attn = attn
        s.time_mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        s.label_emb = nn.Embedding(num_classes + 1, c)
        s.in_conv = nn.Conv2d(3, base, 3, padding=1)
        s.down1 = ResBlock(base, base, c); s.down2 = ResBlock(base, base*2, c); s.down3 = ResBlock(base*2, base*4, c)
        if attn: s.down3_attn = Attn(base*4)
        s.downsample = nn.AvgPool2d(2)
        if attn:
            s.mid1 = ResBlock(base*4, base*4, c); s.mid_attn = Attn(base*4); s.mid2 = ResBlock(base*4, base*4, c)
        else:
            s.mid = ResBlock(base*4, base*4, c)
        s.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        s.up3 = ResBlock(base*4+base*4, base*2, c)
        if attn: s.up3_attn = Attn(base*2)
        s.up2 = ResBlock(base*2+base*2, base, c); s.up1 = ResBlock(base+base, base, c)
        s.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
    def forward(s, x, t, y=None):
        c = s.time_mlp(sinu(t, s.c_dim))
        if y is None: y = torch.full((t.size(0),), s.num_classes, device=t.device, dtype=torch.long)
        c = c + s.label_emb(y)
        x = s.in_conv(x)
        s1 = s.down1(x, c); x = s.downsample(s1)
        s2 = s.down2(x, c); x = s.downsample(s2)
        s3 = s.down3(x, c)
        if s.attn: s3 = s.down3_attn(s3)
        x = s.downsample(s3)
        if s.attn: x = s.mid1(x, c); x = s.mid_attn(x); x = s.mid2(x, c)
        else: x = s.mid(x, c)
        x = s.upsample(x); x = s.up3(torch.cat([x, s3], 1), c)
        if s.attn: x = s.up3_attn(x)
        x = s.upsample(x); x = s.up2(torch.cat([x, s2], 1), c)
        x = s.upsample(x); x = s.up1(torch.cat([x, s1], 1), c)
        return s.out(x)

# ---------- exp06: Stage clip-pooled UNet ----------
class StageC(nn.Module):
    def __init__(s, i, o, c, nb=2, attn=False):
        super().__init__(); s.blocks = nn.ModuleList([ResBlock(i if k == 0 else o, o, c) for k in range(nb)]); s.attn = Attn(o) if attn else None
    def forward(s, x, c):
        for b in s.blocks: x = b(x, c)
        if s.attn is not None: x = s.attn(x)
        return x

class UNetClip(nn.Module):
    def __init__(s, base=128, c=256, nb=2, clip_dim=512, clip_table=None):
        super().__init__(); s.c_dim = c; s.num_classes = clip_table.shape[0]-1
        s.register_buffer("clip_table", clip_table)
        s.text_proj = nn.Sequential(nn.Linear(clip_dim, c), nn.SiLU(), nn.Linear(c, c))
        s.time_mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        s.in_conv = nn.Conv2d(3, base, 3, padding=1)
        s.down1 = StageC(base, base, c, nb); s.down2 = StageC(base, base*2, c, nb); s.down3 = StageC(base*2, base*4, c, nb, attn=True)
        s.downsample = nn.AvgPool2d(2); s.mid = StageC(base*4, base*4, c, nb, attn=True)
        s.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        s.up3 = StageC(base*4+base*4, base*2, c, nb, attn=True); s.up2 = StageC(base*2+base*2, base, c, nb); s.up1 = StageC(base+base, base, c, nb)
        s.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
    def forward(s, x, t, y=None):
        c = s.time_mlp(sinu(t, s.c_dim))
        if y is None: y = torch.full((t.size(0),), s.num_classes, device=t.device, dtype=torch.long)
        c = c + s.text_proj(s.clip_table[y])
        x = s.in_conv(x)
        s1 = s.down1(x, c); x = s.downsample(s1)
        s2 = s.down2(x, c); x = s.downsample(s2)
        s3 = s.down3(x, c); x = s.downsample(s3)
        x = s.mid(x, c)
        x = s.upsample(x); x = s.up3(torch.cat([x, s3], 1), c)
        x = s.upsample(x); x = s.up2(torch.cat([x, s2], 1), c)
        x = s.upsample(x); x = s.up1(torch.cat([x, s1], 1), c)
        return s.out(x)

# ---------- exp08: cross-attn UNet ----------
class StageX(nn.Module):
    def __init__(s, i, o, c, nb=2, self_attn=False, cross_attn=False):
        super().__init__(); s.blocks = nn.ModuleList([ResBlock(i if k == 0 else o, o, c) for k in range(nb)])
        s.self_attn = Attn(o) if self_attn else None; s.cross_attn = CrossAttn(o) if cross_attn else None
    def forward(s, x, c, ctx=None, mask=None):
        for b in s.blocks: x = b(x, c)
        if s.self_attn is not None: x = s.self_attn(x)
        if s.cross_attn is not None: x = s.cross_attn(x, ctx, mask)
        return x

class UNetXattn(nn.Module):
    def __init__(s, base=128, c=256, nb=2, txt=512):
        super().__init__(); s.c_dim = c
        s.text_proj = nn.Sequential(nn.Linear(txt, c), nn.SiLU(), nn.Linear(c, c))
        s.time_mlp = nn.Sequential(nn.Linear(c, c), nn.SiLU(), nn.Linear(c, c))
        s.in_conv = nn.Conv2d(3, base, 3, padding=1)
        s.down1 = StageX(base, base, c, nb); s.down2 = StageX(base, base*2, c, nb, cross_attn=True)
        s.down3 = StageX(base*2, base*4, c, nb, self_attn=True, cross_attn=True)
        s.downsample = nn.AvgPool2d(2); s.mid = StageX(base*4, base*4, c, nb, self_attn=True, cross_attn=True)
        s.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        s.up3 = StageX(base*4+base*4, base*2, c, nb, self_attn=True, cross_attn=True)
        s.up2 = StageX(base*2+base*2, base, c, nb, cross_attn=True); s.up1 = StageX(base+base, base, c, nb)
        s.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), nn.Conv2d(base, 3, 3, padding=1))
    def forward(s, x, t, pooled, tokens, mask):
        c = s.time_mlp(sinu(t, s.c_dim)) + s.text_proj(pooled)
        x = s.in_conv(x)
        s1 = s.down1(x, c); x = s.downsample(s1)
        s2 = s.down2(x, c, tokens, mask); x = s.downsample(s2)
        s3 = s.down3(x, c, tokens, mask); x = s.downsample(s3)
        x = s.mid(x, c, tokens, mask)
        x = s.upsample(x); x = s.up3(torch.cat([x, s3], 1), c, tokens, mask)
        x = s.upsample(x); x = s.up2(torch.cat([x, s2], 1), c, tokens, mask)
        x = s.upsample(x); x = s.up1(torch.cat([x, s1], 1), c)
        return s.out(x)

# ---------- diffusion schedule ----------
beta = torch.linspace(1e-4, 0.02, T, device=dev); alpha = 1 - beta; abar = torch.cumprod(alpha, 0)

@torch.no_grad()
def gen_label(model, x_init, y, guid=GUID):
    n = x_init.size(0); x = x_init.clone(); null = torch.full((n,), model.num_classes, device=dev, dtype=torch.long)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        ec = model(x, t, y=y); eu = model(x, t, y=null); pred = eu + guid*(ec-eu)
        m = (1/alpha[i].sqrt())*(x - (beta[i]/(1-abar[i]).sqrt())*pred)
        x = m + (beta[i].sqrt()*torch.randn_like(x) if i > 0 else 0.0)
    return x

@torch.no_grad()
def gen_xattn(model, x_init, pooled, tokens, mask, null, guid=GUID):
    n = x_init.size(0); x = x_init.clone()
    npn, ntn, nmn = null
    npn = npn[None].expand(n, -1); ntn = ntn[None].expand(n, -1, -1); nmn = nmn[None].expand(n, -1)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev, dtype=torch.long)
        ec = model(x, t, pooled, tokens, mask); eu = model(x, t, npn, ntn, nmn); pred = eu + guid*(ec-eu)
        m = (1/alpha[i].sqrt())*(x - (beta[i]/(1-abar[i]).sqrt())*pred)
        x = m + (beta[i].sqrt()*torch.randn_like(x) if i > 0 else 0.0)
    return x

def save_grid(imgs, names, path, title):
    imgs = ((imgs.clamp(-1, 1)+1)/2).cpu().permute(0, 2, 3, 1).numpy()
    fig, ax = plt.subplots(2, 4, figsize=(11, 6)); fig.suptitle(title, fontsize=15, fontweight="bold")
    for k in range(8):
        a = ax[k//4][k % 4]; a.imshow(imgs[k]); a.axis("off"); a.set_title(names[k], fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(path, dpi=110, bbox_inches="tight"); plt.close()
    print("saved", path)

def find(prefix): return glob.glob(f"{E}/{prefix}*")[0]

# ===== self-attn: exp02 (off) vs exp03 (on) =====
torch.manual_seed(42); X_self = torch.randn(8, 3, 96, 96, device=dev)
cls20 = json.load(open(find("exp02")+"/classes.json"))
y20 = torch.tensor([cls20.index(n) for n in PICK], device=dev)
for tag, prefix, attn in [("OFF (exp02, no self-attn)", "exp02", False), ("ON (exp03, + self-attn)", "exp03", True)]:
    m = UNetFlat(64, num_classes=len(cls20), attn=attn).to(dev)
    m.load_state_dict(torch.load(find(prefix)+"/checkpoints/ckpt_ep300.pt", map_location=dev)); m.eval()
    imgs = gen_label(m, X_self, y20)
    save_grid(imgs, PICK, f"{OUT}/selfattn_{'on' if attn else 'off'}.png", f"Self-Attention {tag}")
    del m; torch.cuda.empty_cache()

# ===== cross-attn: exp06 (off) vs exp08 (on), by name via precomputed feats =====
torch.manual_seed(7); X_cross = torch.randn(8, 3, 96, 96, device=dev)
# exp06
d6 = find("exp06"); ct = torch.load(d6+"/clip_table.pt", map_location=dev)
cls6 = json.load(open(d6+"/classes.json")); y6 = torch.tensor([cls6.index(n) for n in PICK], device=dev)
m6 = UNetClip(128, clip_table=ct).to(dev)
m6.load_state_dict(torch.load(d6+"/checkpoints/ckpt_ep300.pt", map_location=dev)); m6.eval()
imgs = gen_label(m6, X_cross, y6)
save_grid(imgs, PICK, f"{OUT}/crossattn_off.png", "Cross-Attention OFF (exp06, CLIP pooled only)")
del m6; torch.cuda.empty_cache()
# exp08
d8 = find("exp08"); seq = torch.load("/root/autodl-tmp/clip_seq.pt", map_location=dev)
sidx = [seq["names"].index(n) for n in PICK]
pooled = seq["pooled"][sidx].float().to(dev); tokens = seq["tokens"][sidx].float().to(dev); mask = seq["mask"][sidx].to(dev)
null = (seq["null_pooled"].float().to(dev), seq["null_tokens"].float().to(dev), seq["null_mask"].to(dev))
m8 = UNetXattn(128).to(dev)
m8.load_state_dict(torch.load(d8+"/checkpoints/ckpt_ep300.pt", map_location=dev)); m8.eval()
imgs = gen_xattn(m8, X_cross, pooled, tokens, mask, null)
save_grid(imgs, PICK, f"{OUT}/crossattn_on.png", "Cross-Attention ON (exp08, + cross-attn @48)")
print("ALL DONE")
