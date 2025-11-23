"""
Web scraping utilities for fetching app reviews and changelogs.
Respects robots.txt and terms of service.
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from urllib.robotparser import RobotFileParser


# User agent for requests
USER_AGENT = "ProductResearchBot/1.0 (Educational/Research purposes)"


def check_robots_txt(url: str) -> bool:
    """Check if we're allowed to scrape this URL according to robots.txt."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()

        return rp.can_fetch(USER_AGENT, url)
    except Exception:
        # If we can't read robots.txt, be conservative
        return True


def fetch_app_store_reviews(app_id: str, country: str = "us",
                            days_back: int = 90) -> List[Dict]:
    """
    Fetch App Store reviews using public RSS feed.

    Args:
        app_id: The App Store app ID (numeric ID from the URL)
        country: Country code (us, gb, etc.)
        days_back: Only return reviews from the last N days

    Returns:
        List of review dicts with 'title', 'content', 'rating', 'date', 'author'
    """
    reviews = []
    cutoff_date = datetime.now() - timedelta(days=days_back)

    # App Store RSS feed URL (fetches most recent reviews)
    rss_url = f"https://itunes.apple.com/{country}/rss/customerreviews/id={app_id}/sortBy=mostRecent/xml"

    try:
        response = requests.get(rss_url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()

        # Parse XML
        root = ET.fromstring(response.content)

        # Define namespaces
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'im': 'http://itunes.apple.com/rss'
        }

        for entry in root.findall('atom:entry', ns):
            # Skip the first entry which is usually app info
            title_elem = entry.find('atom:title', ns)
            content_elem = entry.find('atom:content', ns)
            author_elem = entry.find('atom:author/atom:name', ns)
            rating_elem = entry.find('im:rating', ns)
            updated_elem = entry.find('atom:updated', ns)

            if content_elem is not None and content_elem.text:
                # Parse date
                review_date = None
                if updated_elem is not None and updated_elem.text:
                    try:
                        review_date = datetime.fromisoformat(updated_elem.text.replace('Z', '+00:00'))
                        if review_date.replace(tzinfo=None) < cutoff_date:
                            continue
                    except ValueError:
                        pass

                reviews.append({
                    'title': title_elem.text if title_elem is not None else '',
                    'content': content_elem.text,
                    'rating': int(rating_elem.text) if rating_elem is not None else None,
                    'date': review_date.isoformat() if review_date else None,
                    'author': author_elem.text if author_elem is not None else 'Anonymous',
                    'source': 'app_store',
                    'source_url': f"https://apps.apple.com/{country}/app/id{app_id}"
                })

    except Exception as e:
        print(f"Error fetching App Store reviews for {app_id}: {e}")

    return reviews


def fetch_play_store_reviews_rss(package_name: str, days_back: int = 90) -> List[Dict]:
    """
    Note: Google Play doesn't have a public RSS feed for reviews.
    This function attempts to get basic app info and suggests alternatives.

    For actual Play Store reviews, users would need to use the official
    Google Play Developer API with proper authentication.

    Returns:
        List of review dicts (may be empty with a note)
    """
    # Google Play doesn't provide public RSS for reviews
    # Return empty list with metadata explaining the limitation
    return [{
        'title': 'Note: Play Store Reviews',
        'content': f'Google Play Store does not provide a public RSS feed for reviews. '
                   f'Consider using the Google Play Developer API for {package_name} '
                   f'or manually exporting reviews from Play Console.',
        'rating': None,
        'date': datetime.now().isoformat(),
        'author': 'System',
        'source': 'play_store_note',
        'source_url': f"https://play.google.com/store/apps/details?id={package_name}"
    }]


def fetch_rss_feed(url: str, days_back: int = 90) -> List[Dict]:
    """
    Fetch and parse a generic RSS/Atom feed.

    Args:
        url: URL of the RSS feed
        days_back: Only return entries from the last N days

    Returns:
        List of entry dicts with 'title', 'content', 'date', 'url'
    """
    entries = []
    cutoff_date = datetime.now() - timedelta(days=days_back)

    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Try Atom format first
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        atom_entries = root.findall('.//atom:entry', ns) or root.findall('.//entry')

        if atom_entries:
            for entry in atom_entries:
                title = entry.find('atom:title', ns) or entry.find('title')
                content = entry.find('atom:content', ns) or entry.find('atom:summary', ns) or \
                          entry.find('content') or entry.find('summary')
                link = entry.find('atom:link', ns) or entry.find('link')
                updated = entry.find('atom:updated', ns) or entry.find('atom:published', ns) or \
                          entry.find('updated') or entry.find('published')

                entry_url = link.get('href') if link is not None else url

                entries.append({
                    'title': title.text if title is not None else '',
                    'content': content.text if content is not None else '',
                    'date': updated.text if updated is not None else None,
                    'url': entry_url,
                    'source': 'rss_feed'
                })
        else:
            # Try RSS 2.0 format
            for item in root.findall('.//item'):
                title = item.find('title')
                description = item.find('description')
                link = item.find('link')
                pub_date = item.find('pubDate')

                entries.append({
                    'title': title.text if title is not None else '',
                    'content': description.text if description is not None else '',
                    'date': pub_date.text if pub_date is not None else None,
                    'url': link.text if link is not None else url,
                    'source': 'rss_feed'
                })

    except Exception as e:
        print(f"Error fetching RSS feed {url}: {e}")

    return entries


def scrape_webpage_content(url: str, respect_robots: bool = True) -> Optional[Dict]:
    """
    Scrape text content from a webpage.

    Args:
        url: URL to scrape
        respect_robots: Whether to check robots.txt first

    Returns:
        Dict with 'title', 'content', 'url' or None if failed/disallowed
    """
    if respect_robots and not check_robots_txt(url):
        print(f"Scraping disallowed by robots.txt: {url}")
        return None

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=30,
            allow_redirects=True
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            element.decompose()

        # Get title
        title = soup.find('title')
        title_text = title.get_text().strip() if title else ''

        # Try to find main content
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=re.compile(r'content|post|entry|article'))

        if main_content:
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Fall back to body content
            body = soup.find('body')
            content = body.get_text(separator=' ', strip=True) if body else ''

        # Clean up content
        content = re.sub(r'\s+', ' ', content)
        content = content[:10000]  # Limit content size

        return {
            'title': title_text,
            'content': content,
            'url': url,
            'source': 'webpage'
        }

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def fetch_changelog_content(url: str, days_back: int = 90) -> List[Dict]:
    """
    Fetch changelog content from a URL.
    Tries RSS first, then falls back to HTML scraping.

    Args:
        url: Changelog/blog URL
        days_back: Time window in days

    Returns:
        List of content dicts
    """
    content_items = []

    # First try to find and fetch RSS feed
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Look for RSS feed link
        rss_link = soup.find('link', type='application/rss+xml') or \
                   soup.find('link', type='application/atom+xml') or \
                   soup.find('a', href=re.compile(r'(rss|feed|atom)', re.I))

        if rss_link:
            rss_url = rss_link.get('href')
            if rss_url and not rss_url.startswith('http'):
                rss_url = urljoin(url, rss_url)
            if rss_url:
                rss_content = fetch_rss_feed(rss_url, days_back)
                if rss_content:
                    return rss_content
    except Exception:
        pass

    # Fall back to scraping the page
    scraped = scrape_webpage_content(url)
    if scraped:
        content_items.append(scraped)

    return content_items


def parse_app_id(input_str: str) -> Tuple[str, str]:
    """
    Parse app ID from various input formats.

    Args:
        input_str: Could be an ID, URL, or package name

    Returns:
        Tuple of (app_id, store_type) where store_type is 'app_store' or 'play_store'
    """
    input_str = input_str.strip()

    # Check for App Store URL
    app_store_match = re.search(r'apps\.apple\.com/\w+/app/[^/]+/id(\d+)', input_str)
    if app_store_match:
        return app_store_match.group(1), 'app_store'

    # Check for Play Store URL
    play_store_match = re.search(r'play\.google\.com/store/apps/details\?id=([^&]+)', input_str)
    if play_store_match:
        return play_store_match.group(1), 'play_store'

    # Check if it's a numeric ID (App Store)
    if input_str.isdigit():
        return input_str, 'app_store'

    # Check if it looks like a package name (Play Store)
    if re.match(r'^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$', input_str, re.I):
        return input_str, 'play_store'

    # Default to treating as App Store ID
    return input_str, 'unknown'
