"""Knowledge-base chunking shared by vector and sparse indexes."""
from __future__ import annotations

from dataclasses import dataclass
import re

CHUNK_SCHEMA_VERSION = 4


@dataclass(frozen=True)
class AnimeChunk:
    anime_id: int
    chunk_type: str
    chunk_index: int
    text: str

    @property
    def id(self) -> str:
        if self.chunk_type == "profile" and self.chunk_index == 0:
            return f"anime_{self.anime_id}"
        return f"anime_{self.anime_id}_{self.chunk_type}_{self.chunk_index}"


def _line(label: str, values) -> str | None:
    if values is None or values == "" or values == []:
        return None
    if isinstance(values, (list, tuple)):
        values = "、".join(str(value).strip() for value in values if str(value).strip())
    return f"{label}: {values}" if values else None


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries, falling back to lossless hard splits."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.findall(r"[^。！？!?；;\n]+[。！？!?；;\n]?", text)
    parts: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                sentence[start:start + max_chars]
                for start in range(0, len(sentence), max_chars)
            )
        elif len(current) + len(sentence) <= max_chars:
            current += sentence
        else:
            parts.append(current)
            current = sentence
    if current:
        parts.append(current)
    return parts


def _single_review_chunks(title: str, comments: list[str], max_chars: int) -> list[str]:
    """Keep reviews isolated and preserve all text when a review needs splitting."""
    prefix = f"番剧: {title}\n观众评论:\n- "
    chunks: list[str] = []

    for raw_comment in comments:
        comment = " ".join(str(raw_comment).split())
        if not comment:
            continue
        available = max(max_chars - len(prefix), 1)
        chunks.extend(prefix + part for part in _split_text(comment, available))
    return chunks


def make_anime_chunks(info: dict, review_chunk_chars: int = 700) -> list[AnimeChunk]:
    """Create field-aware chunks while retaining the anime identity in every chunk."""
    anime_id = int(info["id"])
    title = str(info.get("title") or "未知番剧").strip()
    chunks: list[AnimeChunk] = []

    profile_lines = [
        f"番剧: {title}",
        _line("评分", info.get("score")),
        _line("评分人数", info.get("score_count")),
        _line("播出日期", info.get("date")),
        _line("类型/标签", info.get("tags")),
    ]
    chunks.append(AnimeChunk(
        anime_id=anime_id,
        chunk_type="profile",
        chunk_index=0,
        text="\n".join(line for line in profile_lines if line),
    ))

    staff_lines = [
        f"番剧: {title}",
        _line("制作公司", info.get("studios")),
        _line("导演", info.get("directors")),
        _line("编剧/系列构成", info.get("writers")),
    ]
    if any(staff_lines[1:]):
        chunks.append(AnimeChunk(
            anime_id=anime_id,
            chunk_type="staff",
            chunk_index=0,
            text="\n".join(line for line in staff_lines if line),
        ))

    cast_line = _line("声优", info.get("seiyuu"))
    if cast_line:
        chunks.append(AnimeChunk(
            anime_id=anime_id,
            chunk_type="cast",
            chunk_index=0,
            text=f"番剧: {title}\n{cast_line}",
        ))

    for index, text in enumerate(
        _single_review_chunks(title, info.get("comments", []), review_chunk_chars)
    ):
        chunks.append(AnimeChunk(
            anime_id=anime_id,
            chunk_type="reviews",
            chunk_index=index,
            text=text,
        ))

    return chunks
