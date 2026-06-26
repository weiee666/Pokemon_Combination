# -*- coding: utf-8 -*-
"""
清洗版：抓取 20 只经典宝可梦，只要【正面 + 正常配色 + 全世代(混) + 动画帧】，
去掉【背面 / 闪光异色 / 官方插画 / home 渲染 / dream-world】。不做任何增强。
统一处理成 96x96 白底 RGB -> dataset_prep/pokemon20_clean/<name>__<idx>.png
"""
import os, io, json, hashlib, urllib.request
from PIL import Image, ImageSequence

NAMES = [
    "bulbasaur", "charmander", "squirtle", "venusaur", "charizard", "blastoise",
    "pikachu", "eevee", "jigglypuff", "meowth", "psyduck", "snorlax", "gengar",
    "mewtwo", "mew", "gyarados", "dragonite", "lapras", "magikarp", "aerodactyl",
]

OUT = os.path.join(os.path.dirname(__file__), "pokemon20_clean")
IMG_SIZE = 96
MAX_FRAMES_PER_GIF = 4
RESAMPLE = Image.NEAREST
# 排除：非像素画 + 背面 + 闪光异色
EXCLUDE = ("/other/official-artwork/", "/other/home/",
           "/other/dream-world/", "/other/dream_world/",
           "/back/",        # 背面
           "shiny")          # 闪光异色


def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def collect_urls(sprites):
    urls = set()
    def walk(o):
        if isinstance(o, str):
            low = o.lower()
            if (o.endswith(".png") or o.endswith(".gif")) and not any(x in low for x in EXCLUDE):
                urls.add(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(sprites)
    return urls


def to_square96(img):
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    w, h = img.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), (255, 255, 255))
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    return canvas.resize((IMG_SIZE, IMG_SIZE), RESAMPLE)


def frames_from_gif(data):
    im = Image.open(io.BytesIO(data))
    all_frames = list(ImageSequence.Iterator(im))
    n = len(all_frames)
    if n <= MAX_FRAMES_PER_GIF:
        idxs = range(n)
    else:
        step = n / MAX_FRAMES_PER_GIF
        idxs = [int(i * step) for i in range(MAX_FRAMES_PER_GIF)]
    return [all_frames[i].copy() for i in idxs]


def main():
    os.makedirs(OUT, exist_ok=True)
    grand = 0
    for name in NAMES:
        try:
            data = json.loads(fetch(f"https://pokeapi.co/api/v2/pokemon/{name}"))
        except Exception as e:
            print(f"[{name}] API 失败: {e}")
            continue
        urls = sorted(collect_urls(data["sprites"]))
        seen_hash = set()
        saved = 0
        for url in urls:
            try:
                raw = fetch(url)
            except Exception:
                continue
            try:
                imgs = frames_from_gif(raw) if url.endswith(".gif") else [Image.open(io.BytesIO(raw))]
            except Exception:
                continue
            for im in imgs:
                try:
                    out = to_square96(im)
                except Exception:
                    continue
                h = hashlib.md5(out.tobytes()).hexdigest()
                if h in seen_hash:
                    continue
                seen_hash.add(h)
                out.save(os.path.join(OUT, f"{name}__{saved:04d}.png"))
                saved += 1
        grand += saved
        print(f"[{name:12s}] URL {len(urls):3d} -> 保存 {saved} 张")
    print(f"\n总计: {grand} 张  ->  {OUT}")


if __name__ == "__main__":
    main()
