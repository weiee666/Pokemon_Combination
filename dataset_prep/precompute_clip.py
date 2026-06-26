# -*- coding: utf-8 -*-
"""
离线预计算: 把 descriptions.json 里每只宝可梦的描述用 CLIP 编码成向量,
存成 clip_emb.pt {names, emb(N,512), null(512)}。训练时直接加载, 不再跑 CLIP。
依赖: open_clip_torch。用法: python precompute_clip.py [descriptions.json] [clip_emb.pt]
"""
import sys, json, torch, open_clip

DESC = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/descriptions.json"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/clip_emb.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

descs = json.load(open(DESC, encoding="utf-8"))
names = sorted(descs.keys())

clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tok = open_clip.get_tokenizer("ViT-B-32")
clip = clip.to(device).eval()

@torch.no_grad()
def enc(texts):
    out = []
    for i in range(0, len(texts), 256):
        f = clip.encode_text(tok(texts[i:i+256]).to(device)).float()
        out.append((f / f.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(out, 0)

emb = enc([descs[n] for n in names])
null = enc([""])[0]
torch.save({"names": names, "emb": emb, "null": null}, OUT)
print(f"已存 {OUT}: {len(names)} 只, emb {tuple(emb.shape)}, null {tuple(null.shape)}")
