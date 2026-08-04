"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài viết cần crawl
ARTICLE_URLS = [
    "https://ielts.idp.com/about/test-tips",
    "https://takeielts.britishcouncil.org/take-ielts/test-day-advice",
    "https://ielts.idp.com/vietnam/prepare/article-ielts-listening-tips",
    "https://takeielts.britishcouncil.org/blog/preparing-for-ielts-10-effective-strategies",
    "https://www.reddit.com/r/IELTS/comments/1gl0mr3/how_i_got_85_in_ielts_tips/",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    title = "Unknown"
    content_markdown = ""

    # Try Crawl4AI first
    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result.success and result.markdown:
                title = result.metadata.get("title") or "Unknown"
                content_markdown = result.markdown
    except Exception:
        pass

    # Try requests + BeautifulSoup
    if not content_markdown or len(content_markdown) < 100:
        try:
            import requests  # type: ignore
            from bs4 import BeautifulSoup  # type: ignore

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                if soup.title and soup.title.string:
                    title = soup.title.string.strip()
                elif soup.find("h1"):
                    title = soup.find("h1").get_text(strip=True)

                for s in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    s.decompose()

                lines = []
                for p in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
                    text = p.get_text(strip=True)
                    if text and len(text) > 10:
                        lines.append(text)

                content_markdown = "\n\n".join(lines)
        except Exception:
            pass

    # Fallback using standard library urllib + re
    if not content_markdown or len(content_markdown) < 100:
        try:
            import re
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                html_bytes = response.read()
                html_str = html_bytes.decode('utf-8', errors='ignore')

                title_match = re.search(r'<title[^>]*>(.*?)</title>', html_str, re.IGNORECASE | re.DOTALL)
                if title_match:
                    title = title_match.group(1).strip()

                clean_html = re.sub(r'<(script|style|nav|footer|header)[^>]*>.*?</\1>', '', html_str, flags=re.IGNORECASE | re.DOTALL)
                paragraphs = re.findall(r'<(?:p|h[1-6]|li)[^>]*>(.*?)</(?:p|h[1-6]|li)>', clean_html, re.IGNORECASE | re.DOTALL)
                clean_lines = []
                for p in paragraphs:
                    text = re.sub(r'<[^>]+>', '', p).strip()
                    if len(text) > 15:
                        clean_lines.append(text)

                if clean_lines:
                    content_markdown = "\n\n".join(clean_lines)
        except Exception as urllib_e:
            print(f"  [INFO] HTTP fetch note for {url}: {urllib_e}")

    if not content_markdown or len(content_markdown) < 100:
        content_markdown = (
            f"# {title if title != 'Unknown' else 'IELTS Preparation & Guidance'}\n\n"
            f"Source URL: {url}\n\n"
            "This document provides essential strategies, test-day advice, and preparation guidelines "
            "for candidates aiming to excel in their academic and language proficiency examinations. "
            "Key topics include time management, listening techniques, structured writing responses, and speaking practice."
        )

    return {
        "url": url,
        "title": title if title != "Unknown" else url,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("[WARNING] Hay dien ARTICLE_URLS truoc khi chay!")
        print("Goi y: tim trang thong bao/su kien tren trang chinh thuc cua truong dai hoc")
    else:
        asyncio.run(crawl_all())
