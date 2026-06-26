# -*- coding: utf-8 -*-
"""
把 pokemon_all 里"偏小/不居中"的 sprite 重新取景:
  抠包围盒 -> nearest 放大到填满 ~TARGET/96 -> 居中贴 96x96 白底。
保持像素清晰(NEAREST),不重新下载。输出 pokemon_all_centered/。
"""
import os
import numpy as np
from PIL import Image

SRC = "pokemon_all"
DST = "pokemon_all_centered"
SIZE = 96
TARGET = 84          # 主体长边目标像素(~88% 画幅, 两侧各留 ~6px)
WHITE_TH = 245       # >=该值视为白底

os.makedirs(DST, exist_ok=True)

def recenter(im):
    im = im.convert("RGB")
    a = np.asarray(im)
    mask = (a < WHITE_TH).any(2)              # 非白=内容
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return im.resize((SIZE, SIZE), Image.NEAREST)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = im.crop((x0, y0, x1 + 1, y1 + 1))  # 抠主体
    bw, bh = crop.size
    s = TARGET / max(bw, bh)                   # 等比缩放因子
    nw, nh = max(1, round(bw * s)), max(1, round(bh * s))
    crop = crop.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    canvas.paste(crop, ((SIZE - nw) // 2, (SIZE - nh) // 2))
    return canvas

files = [f for f in os.listdir(SRC) if f.endswith(".png")]
for i, f in enumerate(files):
    recenter(Image.open(os.path.join(SRC, f))).save(os.path.join(DST, f))
    if i % 2000 == 0:
        print(f"  {i}/{len(files)}")
print(f"完成: {len(files)} 张 -> {DST}")
