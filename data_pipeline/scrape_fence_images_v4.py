"""
Automated High-Quality Fence Image Scraper v4.0
================================================
Features:
- NO API keys required (direct web scraping with Playwright)
- Scrapes Pexels, Unsplash, Google Images, Bing Images
- Gemini-powered intelligent query generation
- Perceptual hashing + SHA256 for robust deduplication
- Quality filtering (resolution, aspect ratio, file size)
- Fully automated (no user input)
- Resume capability with persistent cache
- Comprehensive logging and progress tracking

Requirements:
    pip install playwright requests pillow imagehash tqdm python-dotenv google-generativeai
    playwright install chromium

Usage:
    python scrape_fence_images_v4.py

Author: VisionGuard Team
Date: November 10, 2025
"""

import os
import json
import time
import hashlib
import requests
import urllib.parse
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import imagehash
from PIL import Image
import io
from tqdm import tqdm
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import logging
import random

# ============================================================================
# CONFIGURATION
# ============================================================================

# Directories
DOWNLOAD_FOLDER = "./data/images"
METADATA_FILE = "./data/scraping_metadata.json"
HASH_CACHE_FILE = "./data/image_hashes.json"
LOG_FILE = "./data/scraper_v4.log"

# Target collection size
MIN_TOTAL_IMAGES = 2000
MAX_IMAGES_PER_QUERY = 60

# Quality thresholds
MIN_WIDTH = 800
MIN_HEIGHT = 600
MIN_FILE_SIZE = 50 * 1024  # 50 KB
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MIN_ASPECT_RATIO = 0.5
MAX_ASPECT_RATIO = 2.5

# Deduplication thresholds
PERCEPTUAL_HASH_THRESHOLD = 5  # Hamming distance

# Rate limiting
SCROLL_DELAY = 1.5
DOWNLOAD_DELAY = 0.5
MAX_RETRIES = 3

# Threading
MAX_WORKERS = 6

# Blocked domains
BLOCKED_DOMAINS = [
    'istockphoto.com', 'gettyimages.com', 'shutterstock.com',
    'alamy.com', 'dreamstime.com', '123rf.com', 'depositphotos.com',
    'adobestock.com', 'bigstockphoto.com', 'canstockphoto.com'
]

# Fallback fence queries
FENCE_QUERIES_FALLBACK = [
    "wooden fence garden", "picket fence backyard", "chain link fence yard",
    "wrought iron fence residential", "vinyl fence white", "split rail fence farm",
    "privacy fence brown", "wire mesh fence agricultural", "bamboo fence asian",
    "metal fence black", "aluminum fence modern", "composite fence deck",
    "stone pillar fence", "brick fence wall", "glass panel fence contemporary",
    "ranch fence horses", "farm fence cattle", "garden fence vegetable",
    "yard fence boundary", "backyard fence privacy", "front yard fence decorative",
    "suburban fence residential", "rural fence country", "boundary fence property",
    "perimeter fence security", "fence along road", "fence in field green",
    "fence in garden flowers", "fence with gate entry", "modern fence horizontal",
    "traditional fence vertical", "rustic fence weathered", "contemporary fence design",
    "decorative fence ornamental", "lattice fence trellis", "diagonal fence pattern",
    "white picket fence", "black metal fence", "natural wood fence cedar",
    "painted fence colorful", "stained fence dark", "weathered fence aged",
    "fence in sunlight bright", "fence in shadow", "fence at sunset golden",
    "fence in winter snow", "fence in summer flowers", "fence with climbing plants",
    "fence on hillside slope", "fence by lawn grass", "fence near trees woods",
    "colonial fence historic", "craftsman fence style", "victorian fence ornate",
    "mediterranean fence stucco", "japanese fence bamboo", "korean fence traditional",
    "horse fence paddock", "dog fence pet", "pool fence safety",
    "security fence tall", "fence corner angle", "fence post detail closeup"
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_json(filepath: str, default=None):
    """Load JSON file with error handling."""
    if not os.path.exists(filepath):
        return default if default is not None else {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except:
        return default if default is not None else {}


def save_json(filepath: str, data):
    """Save JSON file with error handling."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


# ============================================================================
# PERCEPTUAL HASHING FOR DEDUPLICATION
# ============================================================================

class DualHasher:
    """Manages SHA256 and perceptual hashing for robust deduplication."""
    
    def __init__(self, cache_file: str):
        self.cache_file = cache_file
        self.sha_hashes: Set[str] = set()
        self.perceptual_hashes: Dict[str, str] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load existing hashes from cache."""
        if os.path.exists(self.cache_file):
            try:
                data = load_json(self.cache_file, {})
                self.sha_hashes = set(data.get('sha_hashes', []))
                self.perceptual_hashes = data.get('perceptual_hashes', {})
                logger.info(f"Loaded {len(self.sha_hashes)} SHA + {len(self.perceptual_hashes)} perceptual hashes")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
    
    def _save_cache(self):
        """Save hashes to cache."""
        try:
            save_json(self.cache_file, {
                'sha_hashes': list(self.sha_hashes),
                'perceptual_hashes': self.perceptual_hashes
            })
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def compute_hashes(self, image_bytes: bytes) -> Tuple[str, str]:
        """Compute both SHA256 and perceptual hash."""
        sha_hash = hashlib.sha256(image_bytes).hexdigest()
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            perceptual_hash = str(imagehash.dhash(img))
        except:
            perceptual_hash = None
        
        return sha_hash, perceptual_hash
    
    def is_duplicate(self, image_bytes: bytes, threshold: int = PERCEPTUAL_HASH_THRESHOLD) -> Tuple[bool, str]:
        """Check if image is duplicate."""
        sha_hash, perceptual_hash = self.compute_hashes(image_bytes)
        
        # Exact duplicate
        if sha_hash in self.sha_hashes:
            return True, "exact_duplicate"
        
        # Perceptual duplicate
        if perceptual_hash:
            for existing_hash in self.perceptual_hashes.keys():
                try:
                    hash1 = imagehash.hex_to_hash(perceptual_hash)
                    hash2 = imagehash.hex_to_hash(existing_hash)
                    distance = hash1 - hash2
                    
                    if distance <= threshold:
                        return True, f"similar_dist_{distance}"
                except:
                    continue
        
        return False, "unique"
    
    def add_hashes(self, image_bytes: bytes, filename: str):
        """Add image hashes to tracking system."""
        sha_hash, perceptual_hash = self.compute_hashes(image_bytes)
        
        self.sha_hashes.add(sha_hash)
        if perceptual_hash:
            self.perceptual_hashes[perceptual_hash] = filename
        
        self._save_cache()
    
    def get_count(self) -> int:
        """Get count of unique images."""
        return len(self.sha_hashes)


# ============================================================================
# QUALITY FILTERING
# ============================================================================

def is_high_quality(image_bytes: bytes) -> Tuple[bool, str]:
    """Check if image meets quality standards."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        
        if width < MIN_WIDTH or height < MIN_HEIGHT:
            return False, f"resolution_{width}x{height}"
        
        aspect_ratio = width / height
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return False, f"aspect_{aspect_ratio:.2f}"
        
        file_size = len(image_bytes)
        if file_size < MIN_FILE_SIZE:
            return False, "too_small"
        if file_size > MAX_FILE_SIZE:
            return False, "too_large"
        
        return True, "valid"
    
    except Exception as e:
        return False, f"invalid_{str(e)[:20]}"


def is_blocked_domain(url: str) -> bool:
    """Check if URL is from blocked stock photo site."""
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLOCKED_DOMAINS)


# ============================================================================
# GEMINI QUERY GENERATION
# ============================================================================

def generate_fence_queries_with_gemini(num_queries: int = 50) -> List[str]:
    """Generate diverse fence search queries using Gemini."""
    logger.info(f"Generating {num_queries} search queries with Gemini...")
    
    prompt = f"""Generate {num_queries} diverse, specific search queries for finding high-quality fence images.

REQUIREMENTS:
1. Actual fences only (wooden, vinyl, metal, chain-link, wire, bamboo, composite)
2. Diverse contexts (gardens, lawns, backyards, farms, residential, commercial)
3. Variety in types, materials, styles, colors, settings, weather
4. Fences must be clearly visible and prominent

QUERY STRUCTURE:
Combine [fence type] + [material/color] + [context/setting]

Examples:
- "white vinyl picket fence vegetable garden"
- "weathered cedar privacy fence residential backyard"
- "chain link fence enclosing raised bed garden"
- "black aluminum fence modern front yard"
- "split rail fence pastoral farm field"

AVOID: hedges, walls, plant borders, abstract art

Return as JSON array: {{"queries": ["query1", "query2", ...]}}"""
    
    try:
        schema = {
            "type": "OBJECT",
            "properties": {
                "queries": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                }
            },
            "required": ["queries"]
        }
        
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        config = GenerationConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.9
        )
        
        response = model.generate_content(prompt, generation_config=config)
        data = json.loads(response.text)
        queries = data.get('queries', [])
        
        if queries:
            logger.info(f"✓ Generated {len(queries)} queries with Gemini")
            return queries
    
    except Exception as e:
        logger.warning(f"Gemini query generation failed: {e}")
    
    logger.info("Using fallback queries")
    return FENCE_QUERIES_FALLBACK[:num_queries]


# ============================================================================
# PLAYWRIGHT SCRAPERS
# ============================================================================

def scrape_pexels_playwright(query: str, max_images: int = MAX_IMAGES_PER_QUERY) -> List[str]:
    """Scrape Pexels using Playwright (no API key)."""
    logger.info(f"[Pexels] Searching: '{query}'")
    image_urls = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            # Search on Pexels
            search_url = f"https://www.pexels.com/search/{urllib.parse.quote(query)}/"
            page.goto(search_url, timeout=60000)
            time.sleep(2)
            
            # Scroll to load more images
            for _ in range(5):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_DELAY)
            
            # Extract image URLs from photo cards
            photos = page.query_selector_all('article a.Photo__StyledLink')
            
            for photo in photos[:max_images]:
                try:
                    # Get the link and navigate to photo page
                    href = photo.get_attribute('href')
                    if href and '/photo/' in href:
                        full_url = f"https://www.pexels.com{href}" if href.startswith('/') else href
                        
                        # Open photo page
                        photo_page = browser.new_page()
                        photo_page.goto(full_url, timeout=30000)
                        time.sleep(1)
                        
                        # Find download button or large image
                        img_elem = photo_page.query_selector('img[srcset]')
                        if img_elem:
                            srcset = img_elem.get_attribute('srcset')
                            if srcset:
                                # Get largest version from srcset
                                sources = srcset.split(',')
                                largest = sources[-1].strip().split(' ')[0]
                                if largest and not is_blocked_domain(largest):
                                    image_urls.append(largest)
                        
                        photo_page.close()
                except Exception as e:
                    logger.debug(f"[Pexels] Photo extraction error: {e}")
                    continue
            
            browser.close()
            logger.info(f"[Pexels] Found {len(image_urls)} URLs")
    
    except Exception as e:
        logger.error(f"[Pexels] Error: {e}")
    
    return image_urls


def scrape_unsplash_playwright(query: str, max_images: int = MAX_IMAGES_PER_QUERY) -> List[str]:
    """Scrape Unsplash using Playwright (no API key)."""
    logger.info(f"[Unsplash] Searching: '{query}'")
    image_urls = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            # Search on Unsplash
            search_url = f"https://unsplash.com/s/photos/{urllib.parse.quote(query)}"
            page.goto(search_url, timeout=60000)
            time.sleep(2)
            
            # Scroll to load more images
            for _ in range(5):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_DELAY)
            
            # Extract image URLs
            figures = page.query_selector_all('figure[itemprop="image"]')
            
            for figure in figures[:max_images]:
                try:
                    img = figure.query_selector('img')
                    if img:
                        src = img.get_attribute('src')
                        if src and 'images.unsplash.com' in src:
                            # Get full resolution (remove size parameters)
                            full_url = src.split('?')[0] + '?q=80&w=2000'
                            if not is_blocked_domain(full_url):
                                image_urls.append(full_url)
                except Exception as e:
                    logger.debug(f"[Unsplash] Image extraction error: {e}")
                    continue
            
            browser.close()
            logger.info(f"[Unsplash] Found {len(image_urls)} URLs")
    
    except Exception as e:
        logger.error(f"[Unsplash] Error: {e}")
    
    return image_urls


def scrape_google_images_playwright(query: str, max_images: int = MAX_IMAGES_PER_QUERY) -> List[str]:
    """Scrape Google Images using Playwright."""
    logger.info(f"[Google] Searching: '{query}'")
    image_urls = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()
            
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&tbm=isch"
            page.goto(search_url, timeout=60000)
            
            # Handle consent
            try:
                accept_btn = page.wait_for_selector('button:has-text("Accept all")', timeout=3000)
                if accept_btn:
                    accept_btn.click()
                    time.sleep(1)
            except:
                pass
            
            # Scroll to load images
            for _ in range(6):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_DELAY)
            
            # Extract image URLs
            images = page.query_selector_all('img[src^="http"]')
            
            for img in images[:max_images * 2]:  # Get more to filter
                src = img.get_attribute('src')
                if src and 'gstatic.com' not in src and not is_blocked_domain(src):
                    # Clean URL
                    src_clean = src.split('=w')[0] if '=w' in src else src
                    image_urls.append(src_clean)
            
            browser.close()
            logger.info(f"[Google] Found {len(image_urls)} URLs")
    
    except Exception as e:
        logger.error(f"[Google] Error: {e}")
    
    return image_urls[:max_images]


def scrape_bing_images_playwright(query: str, max_images: int = MAX_IMAGES_PER_QUERY) -> List[str]:
    """Scrape Bing Images using Playwright."""
    logger.info(f"[Bing] Searching: '{query}'")
    image_urls = []
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            
            search_url = f"https://www.bing.com/images/search?q={urllib.parse.quote(query)}"
            page.goto(search_url, timeout=60000)
            time.sleep(2)
            
            # Scroll to load images
            for _ in range(5):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(SCROLL_DELAY)
            
            # Extract from Bing's image grid
            images = page.query_selector_all('a.iusc')
            
            for img in images[:max_images]:
                m_attr = img.get_attribute('m')
                if m_attr:
                    try:
                        data = json.loads(m_attr)
                        url = data.get('murl')
                        if url and not is_blocked_domain(url):
                            image_urls.append(url)
                    except:
                        pass
            
            browser.close()
            logger.info(f"[Bing] Found {len(image_urls)} URLs")
    
    except Exception as e:
        logger.error(f"[Bing] Error: {e}")
    
    return image_urls


# ============================================================================
# IMAGE DOWNLOADER
# ============================================================================

def download_image(url: str, save_path: str, hasher: DualHasher) -> Optional[Dict]:
    """Download and validate image with deduplication."""
    try:
        response = requests.get(url, timeout=15, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code != 200:
            return {"status": "download_failed", "url": url}
        
        image_bytes = response.content
        
        # Quality check
        is_valid, reason = is_high_quality(image_bytes)
        if not is_valid:
            return {"status": reason, "url": url}
        
        # Duplicate check
        is_dup, dup_reason = hasher.is_duplicate(image_bytes)
        if is_dup:
            return {"status": dup_reason, "url": url}
        
        # Save image
        with open(save_path, 'wb') as f:
            f.write(image_bytes)
        
        # Add to hash cache
        hasher.add_hashes(image_bytes, os.path.basename(save_path))
        
        return {
            "status": "success",
            "url": url,
            "path": save_path,
            "size": len(image_bytes)
        }
    
    except Exception as e:
        return {"status": f"error", "url": url}


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    # Load environment
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not found in .env file")
        return
    
    genai.configure(api_key=GEMINI_API_KEY)
    
    # Create directories
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    
    # Initialize hasher
    hasher = DualHasher(HASH_CACHE_FILE)
    
    print("=" * 70)
    print("AUTOMATED FENCE IMAGE SCRAPER v4.0")
    print("=" * 70)
    print(f"Target: {MIN_TOTAL_IMAGES} unique images")
    print(f"Current: {hasher.get_count()} images")
    print(f"Sources: Pexels, Unsplash, Google, Bing (Playwright)")
    print(f"Deduplication: Perceptual hashing + SHA256")
    print("=" * 70)
    
    # Generate queries
    search_queries = generate_fence_queries_with_gemini(num_queries=50)
    
    # Statistics
    stats = {
        "downloaded": 0,
        "duplicates": 0,
        "low_quality": 0,
        "errors": 0
    }
    
    # Process queries
    for query_idx, query in enumerate(search_queries, 1):
        current_count = hasher.get_count()
        
        if current_count >= MIN_TOTAL_IMAGES:
            logger.info(f"✓ Target reached: {current_count} images")
            break
        
        logger.info(f"\n[{query_idx}/{len(search_queries)}] Query: '{query}'")
        
        # Collect URLs from all sources
        all_urls = []
        
        # Randomly select 2-3 sources per query to distribute load
        sources = ['pexels', 'unsplash', 'google', 'bing']
        selected_sources = random.sample(sources, k=random.randint(2, 3))
        
        for source in selected_sources:
            try:
                if source == 'pexels':
                    urls = scrape_pexels_playwright(query)
                elif source == 'unsplash':
                    urls = scrape_unsplash_playwright(query)
                elif source == 'google':
                    urls = scrape_google_images_playwright(query)
                elif source == 'bing':
                    urls = scrape_bing_images_playwright(query)
                else:
                    urls = []
                
                all_urls.extend(urls)
            except Exception as e:
                logger.error(f"  {source} failed: {e}")
        
        logger.info(f"  Total URLs: {len(all_urls)}")
        
        if not all_urls:
            continue
        
        # Download images
        downloaded_this_query = 0
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            
            for url in all_urls[:MAX_IMAGES_PER_QUERY]:
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                filename = f"fence_{query.replace(' ', '_')}_{url_hash}.jpg"
                save_path = os.path.join(DOWNLOAD_FOLDER, filename)
                
                if os.path.exists(save_path):
                    continue
                
                futures.append(executor.submit(download_image, url, save_path, hasher))
            
            for future in as_completed(futures):
                result = future.result()
                
                if result:
                    status = result.get("status")
                    
                    if status == "success":
                        stats["downloaded"] += 1
                        downloaded_this_query += 1
                    elif "duplicate" in status or "similar" in status:
                        stats["duplicates"] += 1
                    elif any(x in status for x in ["resolution", "aspect", "too_"]):
                        stats["low_quality"] += 1
                    else:
                        stats["errors"] += 1
                
                time.sleep(DOWNLOAD_DELAY)
        
        logger.info(f"  Downloaded: {downloaded_this_query} new images")
        time.sleep(2)
    
    # Final summary
    print("\n" + "=" * 70)
    print("SCRAPING COMPLETE")
    print("=" * 70)
    print(f"✓ Downloaded: {stats['downloaded']}")
    print(f"↻ Duplicates: {stats['duplicates']}")
    print(f"⚠ Low quality: {stats['low_quality']}")
    print(f"✗ Errors: {stats['errors']}")
    print(f"\nFinal dataset: {hasher.get_count()} unique images")
    print(f"✓ Images saved to: {DOWNLOAD_FOLDER}")
    print("=" * 70)


if __name__ == '__main__':
    main()
