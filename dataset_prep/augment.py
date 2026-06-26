# -*- coding: utf-8 -*-
"""
对 pokemon20/ 做离线数据增强 -> pokemon20_aug/
增强方式：水平翻转 + 随机平移。
平移约束：平移量严格限制在主体外接框四周的空白边距内，
         保证主体（非白像素）一个都不会被移出 96x96 画面。
"""
import os, glob, random, hashlib
from PIL import Image, ImageChops

SRC = os.path.join(os.path.dirname(__file__), "pokemon20")
DST = os.path.join(os.path.dirname(__file__), "pokemon20_aug")
SIZE = 96
N_TRANS = 2          # 每张(原图/翻转)各生成几个随机平移版本
WHITE = (255, 255, 255)
random.seed(0)


def content_bbox(img):
    """返回非白主体的外接框 (l, t, r, b)；全白则 None。"""
    bg = Image.new("RGB", img.size, WHITE)
    return ImageChops.difference(img, bg).getbbox()


def random_shift(img, bbox):
    """在不让主体出界的前提下随机平移；放不下就返回 None。"""
    l, t, r, b = bbox
    left_room, top_room = l, t                 # 可向左/上移的最大量
    right_room, bottom_room = SIZE - r, SIZE - b  # 可向右/下移的最大量
    if left_room + right_room == 0 and top_room + bottom_room == 0:
        return None                            # 主体占满画面，无空间
    for _ in range(5):                         # 尝试几次，避免抽到 (0,0)
        dx = random.randint(-left_room, right_room)
        dy = random.randint(-top_room, bottom_room)
        if dx != 0 or dy != 0:
            break
    else:
        return None
    canvas = Image.new("RGB", (SIZE, SIZE), WHITE)
    canvas.paste(img, (dx, dy))                # 越界部分(白边)被裁掉，主体保留
    return canvas


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    seen = set()
    total = 0

    def save(img, name):
        nonlocal total
        h = hashlib.md5(img.tobytes()).hexdigest()
        if h in seen:                          # 去掉完全相同的(如无空间平移=原图)
            return
        seen.add(h)
        img.save(os.path.join(DST, name))
        total += 1

    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]   # 形如 pikachu__0007
        img = Image.open(f).convert("RGB")
        flip = img.transpose(Image.FLIP_LEFT_RIGHT)

        save(img,  f"{stem}.png")              # 原图
        save(flip, f"{stem}_f.png")            # 水平翻转

        bbox, bbox_f = content_bbox(img), content_bbox(flip)
        if bbox:
            for i in range(N_TRANS):
                s = random_shift(img, bbox)
                if s: save(s, f"{stem}_t{i}.png")
        if bbox_f:
            for i in range(N_TRANS):
                s = random_shift(flip, bbox_f)
                if s: save(s, f"{stem}_ft{i}.png")

    # 每类统计
    import collections
    cnt = collections.Counter(
        os.path.basename(p).split("__")[0] for p in glob.glob(os.path.join(DST, "*.png")))
    print("增强后每类:", dict(sorted(cnt.items())))
    print(f"\n原始 {len(files)} 张 -> 增强后 {total} 张  ->  {DST}")


if __name__ == "__main__":
    main()
