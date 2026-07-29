"""Probe Bangumi subject infobox to design the alias parser."""
import re
import sys
import urllib.request

HDRS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
PROXY = "http://127.0.0.1:7897"
_handler = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
_opener = urllib.request.build_opener(_handler)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=HDRS)
    with _opener.open(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


# 覆盖 seed 里的高频简称。挑几部预期一定有别名的
SAMPLE_IDS = [
    265,     # 新世纪福音战士 EVA
    10380,   # 命运石之门 石头门
    120925,  # 为美好的世界献上祝福！ 素晴
    259369,  # (小说) 无职转生
    296317,  # 无职转生 TV
    26302,   # Re:从零开始的异世界生活
    38124,   # 进击的巨人
    315,     # 钢之炼金术师
]


# 提取 infobox 里所有 <li> 的 (tip, value_text) 键值对
LI_RE = re.compile(
    r'<li[^>]*>\s*<span[^>]*class="tip"[^>]*>([^<]*)</span>\s*(.*?)</li>',
    re.DOTALL,
)
TAG_RE = re.compile(r"<[^>]+>")

ALIAS_KEYS = ("别名", "第二中文名", "英文名", "日文名", "其他", "其他名")
NAME_KEYS = ("中文名",)


def parse_infobox(html: str) -> list[tuple[str, str]]:
    """Return list of (key, value_text) pairs found under #infobox (main + sub_container)."""
    # 主 infobox：允许嵌套 <ul>，用宽松匹配到最靠后的 </ul>
    m = re.search(r'<ul id="infobox">(.*?)</ul>\s*</div>', html, re.DOTALL)
    body = m.group(1) if m else ""
    if not body:
        m = re.search(r'<ul id="infobox">(.*)', html, re.DOTALL)
        body = m.group(1) if m else ""

    pairs = []
    seen = set()
    for m in LI_RE.finditer(body):
        key = m.group(1).replace(":", "").replace("：", "").strip()
        raw_val = m.group(2)
        text = TAG_RE.sub("", raw_val)
        text = re.sub(r"\s+", " ", text).strip()
        sig = (key, text)
        if sig in seen:
            continue
        seen.add(sig)
        pairs.append((key, text))
    return pairs


def extract_aliases(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in pairs:
        if not v:
            continue
        if k in ALIAS_KEYS or k in NAME_KEYS:
            if k in NAME_KEYS:
                # 中文名字段本身就是单一值，不 split（Bangumi 里可能带空格）
                out.setdefault(k, []).append(v.strip())
            else:
                # 别名等多值字段用常见分隔符切
                parts = re.split(r"[\/|、;；,，]+", v)
                parts = [p.strip() for p in parts if p.strip()]
                out.setdefault(k, []).extend(parts)
    return out


for sid in SAMPLE_IDS:
    url = f"https://bangumi.tv/subject/{sid}"
    try:
        html = fetch(url)
    except Exception as e:
        print(f"[{sid}] fetch failed: {e}")
        continue
    pairs = parse_infobox(html)
    aliases = extract_aliases(pairs)
    print(f"=== [{sid}] {url} ===")
    if aliases:
        for k, vs in aliases.items():
            print(f"  {k}: {vs}")
    else:
        print("  (no alias-like fields)")
    print()
