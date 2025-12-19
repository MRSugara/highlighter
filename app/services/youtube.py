import re
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """
    Ekstrak YouTube video ID dari berbagai format URL.

    Mendukung:
    - https://www.youtube.com/watch?v=VIDEOID
    - https://youtu.be/VIDEOID
    - https://m.youtube.com/watch?v=VIDEOID
    - https://www.youtube.com/shorts/VIDEOID
    - https://www.youtube.com/embed/VIDEOID
    """
    if not url:
        raise ValueError("URL YouTube kosong")

    url = url.strip()
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    # watch?v=VIDEOID
    if host.endswith("youtube.com"):
        vid = parse_qs(parsed.query).get("v", [None])[0]
        if vid:
            return vid
        # /shorts/VIDEOID or /embed/VIDEOID or /v/VIDEOID
        m = re.search(r"/(shorts|embed|v)/([0-9A-Za-z_-]{11})", parsed.path or "")
        if m:
            return m.group(2)

    # youtu.be/VIDEOID
    if host == "youtu.be":
        # Path might be '/VIDEOID'
        path_id = (parsed.path or "").lstrip("/")
        if re.fullmatch(r"[0-9A-Za-z_-]{11}", path_id or ""):
            return path_id

    # Fallback: try to find 11-char ID anywhere in the string
    m = re.search(r"([0-9A-Za-z_-]{11})", url)
    if m:
        return m.group(1)

    raise ValueError("URL YouTube tidak valid atau ID tidak ditemukan")


def get_video_id(url: str) -> str:
    """Alias yang digunakan oleh route untuk mengambil video ID."""
    return extract_video_id(url)
