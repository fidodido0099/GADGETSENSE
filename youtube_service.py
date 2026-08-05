
import asyncio
import html
import logging
import os
import re
from dataclasses import dataclass, field
from urllib.parse import quote_plus, urlencode

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Piped API instances (rotated on failure)
# ---------------------------------------------------------------------------
PIPED_INSTANCES = [
    
    "https://api.piped.private.coffee",
    "https://pipedapi.kavin.rocks",
    "https://pipedapi-libre.kavin.rocks",
    "https://pipedapi.leptons.xyz",
    "https://pipedapi.nosebs.ru",
    "https://piped-api.privacy.com.de",
    "https://pipedapi.adminforge.de",
    "https://api.piped.yt",
    "https://pipedapi.drgns.space",
    "https://pipedapi.owo.si",
    "https://pipedapi.ducks.party",
    "https://piped-api.codespace.cz",
    "https://pipedapi.reallyaweso.me",
    "https://pipedapi.darkness.services",
    "https://pipedapi.orangenet.cc",
]

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
REQUEST_TIMEOUT = 15.0
MAX_COMMENT_PAGES = 10  # safety cap on pagination


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class VideoInfo:
    video_id: str
    title: str
    channel: str
    views: int
    thumbnail: str
    url: str


@dataclass
class CommentData:
    text: str
    author: str
    likes: int = 0
    video_id: str = ""


@dataclass
class VideoWithComments:
    video: VideoInfo
    comments: list[CommentData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Comment cleaning
# ---------------------------------------------------------------------------
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+")
_EMOJI_ONLY_RE = re.compile(
    r"^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
    r"\U0001F1E0-\U0001F1FF\U00002700-\U000027BF\U0000FE00-\U0000FE0F"
    r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\s]+$"
)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_comment(raw: str) -> str | None:
    """Clean a single comment. Returns None if it should be filtered out."""
    if _URL_RE.search(raw):
        return None

    text = _HTML_TAG_RE.sub("", raw)
    text = html.unescape(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()

    # Filter rules
    if len(text) < 5 or len(text) > 500:
        return None
    if len(text.split()) <= 2:
        return None
    if _EMOJI_ONLY_RE.match(text):
        return None
    return text


def deduplicate_comments(comments: list[CommentData]) -> list[CommentData]:
    """Remove exact and near-duplicate comments."""
    seen: set[str] = set()
    unique: list[CommentData] = []
    for c in comments:
        key = c.text.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def clean_and_filter(comments: list[CommentData]) -> list[CommentData]:
    """Full cleaning pipeline."""
    cleaned: list[CommentData] = []
    for c in comments:
        text = clean_comment(c.text)
        if text:
            cleaned.append(CommentData(text=text, author=c.author, likes=c.likes, video_id=c.video_id))
    return deduplicate_comments(cleaned)


# ---------------------------------------------------------------------------
# Piped API client
# ---------------------------------------------------------------------------
class PipedClient:
    """Fetches videos and comments from Piped API with multi-instance failover."""

    def __init__(self):
        self._instances = list(PIPED_INSTANCES)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)
        return self._client

    async def _request(self, path: str, params: dict | None = None) -> dict | list | None:
        """Try each instance in order; return first successful JSON response."""
        client = await self._get_client()
        errors: list[str] = []
        for base in self._instances:
            url = f"{base}{path}"
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    body = resp.text.strip()
                    if not body:
                        errors.append(f"{base}: empty response body")
                        continue
                    data = resp.json()
                    # Validate we got meaningful data
                    if isinstance(data, dict) and data.get("error"):
                        errors.append(f"{base}: API error: {data['error']}")
                        continue
                    return data
                errors.append(f"{base}: HTTP {resp.status_code}")
            except Exception as exc:
                errors.append(f"{base}: {type(exc).__name__}: {str(exc)[:80]}")
        logger.warning("All Piped instances failed for %s: %s", path, "; ".join(errors))
        return None

    async def search_videos(self, query: str, limit: int = 5) -> list[VideoInfo]:
        """Search for review videos. Returns up to `limit` results."""
        data = await self._request("/search", params={"q": f"{query} review", "filter": "videos"})
        if not data:
            return []

        # Response: {"items": [...], "nextpage": ..., "suggestion": ..., "corrected": ...}
        if isinstance(data, dict):
            items = data.get("items", [])
        elif isinstance(data, list):
            items = data
        else:
            return []

        if not items:
            return []

        videos: list[VideoInfo] = []
        for item in items:
            if len(videos) >= limit:
                break
            # Filter to stream items only (skip channels, playlists)
            if isinstance(item, dict) and item.get("type", "stream") != "stream":
                continue
            # Skip YouTube Shorts
            if item.get("isShort", False):
                continue

            vid_url = item.get("url", "")
            # Extract video ID from URL like "/watch?v=XXXXX"
            vid_id = ""
            if "v=" in vid_url:
                vid_id = vid_url.split("v=")[-1].split("&")[0]
            elif vid_url.startswith("/watch?"):
                # Fallback: try to parse from query string
                for part in vid_url.split("?")[1].split("&"):
                    if part.startswith("v="):
                        vid_id = part[2:]
                        break

            if not vid_id:
                continue

            videos.append(
                VideoInfo(
                    video_id=vid_id,
                    title=item.get("title", "Unknown"),
                    channel=item.get("uploaderName", "Unknown"),
                    views=item.get("views", 0),
                    thumbnail=item.get("thumbnail", ""),
                    url=f"https://youtube.com/watch?v={vid_id}",
                )
            )
        return videos

    async def get_video_info(self, video_id: str) -> VideoInfo | None:
        """Get detailed info for a single video."""
        data = await self._request(f"/streams/{video_id}")
        if not data or not isinstance(data, dict):
            return None
        return VideoInfo(
            video_id=video_id,
            title=data.get("title", "Unknown"),
            channel=data.get("uploader", "Unknown"),
            views=data.get("views", 0),
            thumbnail=data.get("thumbnailUrl", ""),
            url=f"https://youtube.com/watch?v={video_id}",
        )

    async def get_comments(self, video_id: str, max_pages: int = MAX_COMMENT_PAGES) -> list[CommentData]:
        """Fetch comments with pagination."""
        all_comments: list[CommentData] = []

        data = await self._request(f"/comments/{video_id}")
        if not data or not isinstance(data, dict):
            return all_comments

        self._extract_comments(data, video_id, all_comments)
        nextpage = data.get("nextpage")
        page = 1

        while nextpage and page < max_pages:
            data = await self._request(
                f"/nextpage/comments/{video_id}",
                params={"nextpage": nextpage},
            )
            if not data or not isinstance(data, dict):
                break
            self._extract_comments(data, video_id, all_comments)
            nextpage = data.get("nextpage")
            page += 1

        return all_comments

    @staticmethod
    def _extract_comments(data: dict, video_id: str, out: list[CommentData]):
        for c in data.get("comments", []):
            text = c.get("commentText", "")
            if text:
                out.append(
                    CommentData(
                        text=text,
                        author=c.get("author", "Anonymous"),
                        likes=c.get("likeCount", 0),
                        video_id=video_id,
                    )
                )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# YouTube Data API v3 fallback
# ---------------------------------------------------------------------------
class YouTubeV3Client:
    """Fallback client using YouTube Data API v3 (requires YOUTUBE_API_KEY env var)."""

    def __init__(self):
        self.api_key = os.environ.get("YOUTUBE_API_KEY", "")
        self._client: httpx.AsyncClient | None = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)
        return self._client

    async def search_videos(self, query: str, limit: int = 5) -> list[VideoInfo]:
        if not self.available:
            return []
        client = await self._get_client()
        params = {
            "part": "snippet",
            "q": f"{query} review",
            "type": "video",
            "maxResults": limit,
            "key": self.api_key,
        }
        try:
            resp = await client.get(f"{YOUTUBE_API_BASE}/search", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.error("YouTube v3 search failed: %s", exc)
            return []

        video_ids = [item["id"]["videoId"] for item in data.get("items", []) if "videoId" in item.get("id", {})]
        if not video_ids:
            return []

        # Fetch view counts
        stats = await self._get_video_stats(video_ids)

        videos: list[VideoInfo] = []
        for item in data.get("items", []):
            vid_id = item.get("id", {}).get("videoId", "")
            if not vid_id:
                continue
            snippet = item.get("snippet", {})
            videos.append(
                VideoInfo(
                    video_id=vid_id,
                    title=snippet.get("title", "Unknown"),
                    channel=snippet.get("channelTitle", "Unknown"),
                    views=stats.get(vid_id, 0),
                    thumbnail=snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                    url=f"https://youtube.com/watch?v={vid_id}",
                )
            )
        return videos

    async def _get_video_stats(self, video_ids: list[str]) -> dict[str, int]:
        client = await self._get_client()
        params = {
            "part": "statistics",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        try:
            resp = await client.get(f"{YOUTUBE_API_BASE}/videos", params=params)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            return {}
        result: dict[str, int] = {}
        for item in data.get("items", []):
            vid_id = item.get("id", "")
            views = int(item.get("statistics", {}).get("viewCount", 0))
            result[vid_id] = views
        return result

    async def get_comments(self, video_id: str, max_results: int = 100) -> list[CommentData]:
        if not self.available:
            return []
        client = await self._get_client()
        all_comments: list[CommentData] = []
        next_page_token = None
        fetched = 0

        while fetched < max_results:
            params: dict = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_results - fetched),
                "textFormat": "plainText",
                "key": self.api_key,
            }
            if next_page_token:
                params["pageToken"] = next_page_token

            try:
                resp = await client.get(f"{YOUTUBE_API_BASE}/commentThreads", params=params)
                if resp.status_code == 403:
                    logger.warning("Comments disabled for video %s", video_id)
                    break
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("YouTube v3 comments failed for %s: %s", video_id, exc)
                break

            for item in data.get("items", []):
                snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
                text = snippet.get("textDisplay", "")
                if text:
                    all_comments.append(
                        CommentData(
                            text=text,
                            author=snippet.get("authorDisplayName", "Anonymous"),
                            likes=snippet.get("likeCount", 0),
                            video_id=video_id,
                        )
                    )
                    fetched += 1

            next_page_token = data.get("nextPageToken")
            if not next_page_token:
                break

        return all_comments

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


# ---------------------------------------------------------------------------
# Unified service
# ---------------------------------------------------------------------------
class YouTubeService:
    """Unified service: tries Piped first, falls back to YouTube Data API v3."""

    def __init__(self):
        self.piped = PipedClient()
        self.youtube_v3 = YouTubeV3Client()

    async def search_and_collect(
        self, gadget: str, video_limit: int = 5, min_comments: int = 500
    ) -> list[VideoWithComments]:
        """
        Search for review videos, collect comments from each, clean and return.
        Targets `min_comments` total across all videos.
        """
        # 1. Search for videos
        videos = await self.piped.search_videos(gadget, limit=video_limit)
        source = "piped"

        if not videos:
            logger.info("Piped search returned no results, trying YouTube v3 fallback")
            videos = await self.youtube_v3.search_videos(gadget, limit=video_limit)
            source = "youtube_v3"

        if not videos:
            logger.warning("No videos found for query: %s", gadget)
            return []

        logger.info("Found %d videos via %s for '%s'", len(videos), source, gadget)

        # 2. Enrich video info if views are missing (Piped search doesn't always return views)
        for i, v in enumerate(videos):
            if v.views == 0 and source == "piped":
                info = await self.piped.get_video_info(v.video_id)
                if info:
                    videos[i] = info

        # 3. Collect comments from each video
        results: list[VideoWithComments] = []
        total_comments = 0

        for video in videos:
            if source == "piped":
                raw_comments = await self.piped.get_comments(video.video_id)
            else:
                raw_comments = await self.youtube_v3.get_comments(video.video_id, max_results=150)

            # If Piped returned nothing, try YouTube v3 for this specific video
            if not raw_comments and source == "piped" and self.youtube_v3.available:
                logger.info("Piped comments empty for %s, trying v3 fallback", video.video_id)
                raw_comments = await self.youtube_v3.get_comments(video.video_id, max_results=150)

            cleaned = clean_and_filter(raw_comments)
            results.append(VideoWithComments(video=video, comments=cleaned))
            total_comments += len(cleaned)
            logger.info(
                "Video '%s': %d raw → %d cleaned comments",
                video.title[:50],
                len(raw_comments),
                len(cleaned),
            )

        logger.info("Total cleaned comments collected: %d (target: %d)", total_comments, min_comments)
        return results

    async def close(self):
        await self.piped.close()
        await self.youtube_v3.close()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Test comment cleaning
    test_cases = [
        ("<b>Great product!</b> Love it", "Great product! Love it"),
        ("Check https://spam.com out", "Check out"),
        ("Hi", None),  # too short
        ("😀🎉✨", None),  # emoji-only
        ("A" * 501, None),  # too long
        ("  Normal   comment   here  ", "Normal comment here"),
    ]
    print("Comment cleaning tests:")
    for raw, expected in test_cases:
        result = clean_comment(raw)
        status = "✅" if result == expected else "❌"
        print(f"  {status} clean_comment({raw!r:.40}) → {result!r} (expected {expected!r})")

    # Test deduplication
    dupes = [
        CommentData(text="awesome phone", author="A"),
        CommentData(text="Awesome Phone", author="B"),
        CommentData(text="totally different", author="C"),
    ]
    deduped = deduplicate_comments(dupes)
    print(f"\n  Dedup: {len(dupes)} → {len(deduped)} (expected 2) {'✅' if len(deduped) == 2 else '❌'}")
    print("\nAll cleaning tests complete.")
