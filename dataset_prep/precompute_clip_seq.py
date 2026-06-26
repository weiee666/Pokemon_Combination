# -*- coding: utf-8 -*-
"""
离线预计算【逐 token】CLIP 文本特征(给 cross-attention 用)。
存 clip_seq.pt {names, tokens(N,77,512) fp16, pooled(N,512), mask(N,77), null_*}。
依赖 open_clip_torch。用法: python precompute_clip_seq.py [descriptions.json] [clip_seq.pt]
"""
import sys, json, torch, open_clip
DESC = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/descriptions.json"
OUT  = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/clip_seq.pt"
dev = "cuda" if torch.cuda.is_available() else "cpu"

descs = json.load(open(DESC, encoding="utf-8")); names = sorted(descs.keys())
clip, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
tk = open_clip.get_tokenizer("ViT-B-32"); clip = clip.to(dev).eval()

@torch.no_grad()
def enc(texts):
    toks, pools, masks = [], [], []
    for i in range(0, len(texts), 128):
        t = tk(texts[i:i+128]).to(dev)
        x = clip.token_embedding(t) + clip.positional_embedding   # batch_first(open_clip 3.x), 不要 permute
        x = clip.transformer(x, attn_mask=clip.attn_mask)
        x = clip.ln_final(x)                                       # (b,77,512) 逐 token
        p = x[torch.arange(x.shape[0]), t.argmax(-1)] @ clip.text_projection
        p = p / p.norm(dim=-1, keepdim=True)
        toks.append(x.half().cpu()); pools.append(p.float().cpu()); masks.append((t == 0).cpu())
    return torch.cat(toks), torch.cat(pools), torch.cat(masks)

tok, pool, mask = enc([descs[n] for n in names])
ntok, npool, nmask = enc([""])
torch.save({"names": names, "tokens": tok, "pooled": pool, "mask": mask,
            "null_tokens": ntok[0], "null_pooled": npool[0], "null_mask": nmask[0]}, OUT)
print(f"已存 {OUT}: {len(names)} 只 | tokens {tuple(tok.shape)} {tok.dtype}")
