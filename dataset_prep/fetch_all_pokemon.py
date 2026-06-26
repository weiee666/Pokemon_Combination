# -*- coding: utf-8 -*-
"""
抓取【全世代所有宝可梦】(1025 只) 的像素 sprite + 生成带语义的文字描述。
- 只要正面 + 正常配色 + 像素 sprite(排除背面/异色/官方插画/home/dream-world)
- 描述含: 颜色 / 属性 / 类别 / 体型 / 世代
- 输出: dataset_prep/pokemon_all/<name>__<idx>.png  +  dataset_prep/descriptions.json
- 可断点续传: 已有图片的宝可梦会跳过
"""
import urllib.request, json, io, os, hashlib, time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageSequence

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "pokemon_all")
DESC_PATH = os.path.join(HERE, "descriptions.json")
IMG_SIZE = 96
MAX_FRAMES_PER_GIF = 3
RESAMPLE = Image.NEAREST
EXCLUDE = ("/other/official-artwork/", "/other/home/",
           "/other/dream-world/", "/other/dream_world/", "/back/", "shiny")
ROMANS = {"generation-i":1,"generation-ii":2,"generation-iii":3,"generation-iv":4,
          "generation-v":5,"generation-vi":6,"generation-vii":7,"generation-viii":8,"generation-ix":9}


def _open(u):
    return urllib.request.urlopen(
        urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=25).read()

def get_json(u):
    for a in range(5):                       # 重试, 扛瞬时网络重置
        try:
            return json.loads(_open(u))
        except Exception:
            if a == 4: raise
            time.sleep(1.5 * (a + 1))

def get_bytes(u):
    for a in range(4):
        try:
            return _open(u)
        except Exception:
            if a == 3: raise
            time.sleep(1.0 * (a + 1))


def collect_urls(sprites):
    urls = set()
    def walk(o):
        if isinstance(o, str):
            low = o.lower()
            if (o.endswith(".png") or o.endswith(".gif")) and not any(x in low for x in EXCLUDE):
                urls.add(o)
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(sprites)
    return sorted(urls)


def to_square96(img):
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (255,)*4), img).convert("RGB")
    else:
        img = img.convert("RGB")
    w, h = img.size; s = max(w, h)
    c = Image.new("RGB", (s, s), (255, 255, 255)); c.paste(img, ((s-w)//2, (s-h)//2))
    return c.resize((IMG_SIZE, IMG_SIZE), RESAMPLE)


def frames_from_gif(data):
    im = Image.open(io.BytesIO(data)); fr = list(ImageSequence.Iterator(im)); n = len(fr)
    if n <= MAX_FRAMES_PER_GIF:
        idxs = range(n)
    else:
        step = n / MAX_FRAMES_PER_GIF; idxs = [int(i*step) for i in range(MAX_FRAMES_PER_GIF)]
    return [fr[i].copy() for i in idxs]


def build_desc(sp, pk):
    gen = ROMANS.get(sp["generation"]["name"], "?")
    color = sp["color"]["name"]
    shape = (sp.get("shape") or {}).get("name", "")
    genus = next((g["genus"] for g in sp["genera"] if g["language"]["name"] == "en"), "pokémon")
    types = [t["type"]["name"] for t in pk["types"]]
    pretty = pk["name"].replace("-", " ").title()          # 名字作为 CLIP 的身份锚点
    return f"{pretty}, a {color} {'/'.join(types)}-type {genus.lower()}, {shape} body, from generation {gen}"


def main():
    os.makedirs(OUT, exist_ok=True)
    descs = json.load(open(DESC_PATH, encoding="utf-8")) if os.path.exists(DESC_PATH) else {}
    existing = {f.split("__")[0] for f in os.listdir(OUT)}
    species = get_json("https://pokeapi.co/api/v2/pokemon-species?limit=2000")["results"]
    print(f"共 {len(species)} 只, 已完成 {len(existing)} 只, 开始...")

    grand = sum(1 for _ in os.listdir(OUT))
    for k, s in enumerate(species):
        name = s["name"]
        if name in existing and name in descs:
            continue
        try:
            sp = get_json(s["url"])
            pk = get_json(f"https://pokeapi.co/api/v2/pokemon/{name}")
        except Exception:
            continue
        try:
            descs[name] = build_desc(sp, pk)
        except Exception:
            continue
        def fetch_one(url):
            try:
                raw = get_bytes(url)
                return frames_from_gif(raw) if url.endswith(".gif") else [Image.open(io.BytesIO(raw))]
            except Exception:
                return []
        urls = collect_urls(pk["sprites"])
        with ThreadPoolExecutor(max_workers=16) as ex:    # 并行下载这只的所有图
            batches = list(ex.map(fetch_one, urls))

        seen = set(); saved = 0
        for imgs in batches:
            for im in imgs:
                try:
                    out = to_square96(im)
                except Exception:
                    continue
                h = hashlib.md5(out.tobytes()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                out.save(os.path.join(OUT, f"{name}__{saved:04d}.png")); saved += 1
        grand += saved
        if k % 50 == 0:
            json.dump(descs, open(DESC_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  [{k+1}/{len(species)}] {name}: +{saved}  累计图 {grand}")

    json.dump(descs, open(DESC_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n完成. 描述 {len(descs)} 条, 图片 {grand} 张 -> {OUT}")


if __name__ == "__main__":
    main()
