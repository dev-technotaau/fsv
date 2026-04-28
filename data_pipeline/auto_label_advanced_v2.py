"""
Advanced Auto-Labeling with Polygon Refinement v2.0
===================================================
Upgrades:
- Shapely-based polygon simplification and merging
- Minimum area filtering to remove noise
- Overlapping polygon union
- Edge smoothing with Douglas-Peucker algorithm
- Better error handling and retry logic
- Multi-threaded processing with progress tracking

Research basis:
- Polygon simplification: Douglas-Peucker (1973)
- Morphological operations: Serra (1982)
- Union operations: Shapely computational geometry
"""

import os
import json
import glob
import logging
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import validation

# ============================================================================
# CONFIGURATION
# ============================================================================
IMAGE_DIR = 'data/images'
NO_FENCE_DIR = 'data/no_fence_images'
MAX_WORKERS = 10
LOG_LEVEL = logging.INFO

# Polygon refinement parameters
MIN_AREA_THRESHOLD = 500  # pixels² - ignore tiny detections
SIMPLIFY_TOLERANCE = 2.0  # Douglas-Peucker tolerance (higher = simpler)
MIN_POLYGON_POINTS = 3
MAX_POLYGON_POINTS = 500

# ============================================================================
# SETUP LOGGING
# ============================================================================
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_label_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD ENVIRONMENT & CONFIGURE API
# ============================================================================
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')

if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment!")
    exit(1)

genai.configure(api_key=API_KEY)

# Enhanced prompt with better instructions
PROMPT = """
Analyze this image and detect all fences with PRECISE BOUNDARIES.

Return a JSON object with these keys:
1. "has_fence": boolean (true if ANY fence exists, false otherwise)
2. "polygons": list of polygons, where each polygon is a list of [y, x] coordinates in NORMALIZED format (0.0 to 1.0)

**CRITICAL INSTRUCTIONS:**
- Trace fence boundaries PRECISELY, following the actual fence edges
- Include ALL visible fence segments (posts, rails, pickets, wire mesh)
- Use sufficient points to capture curves and angles (10-50 points typical)
- Normalize coordinates: divide by image height for y, image width for x
- Order points clockwise or counter-clockwise around the fence perimeter
- If multiple fence segments exist, create separate polygons for each
- Ignore shadows, reflections, or background objects

**Example output:**
{
  "has_fence": true,
  "polygons": [
    [[0.1, 0.2], [0.1, 0.8], [0.3, 0.8], [0.3, 0.2]],
    [[0.5, 0.3], [0.5, 0.7], [0.6, 0.7], [0.6, 0.3]]
  ]
}

If NO fence exists in the image:
{
  "has_fence": false,
  "polygons": []
}
"""

# ============================================================================
# POLYGON REFINEMENT FUNCTIONS
# ============================================================================

def refine_polygons_advanced(polygons_raw, img_width, img_height):
    """
    Advanced polygon refinement using computational geometry.
    
    Steps:
    1. Convert normalized coords to pixels
    2. Create Shapely polygons
    3. Filter by minimum area threshold
    4. Simplify using Douglas-Peucker algorithm
    5. Merge overlapping polygons
    6. Convert back to normalized coords
    
    Args:
        polygons_raw: List of polygons with normalized [y, x] coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        List of refined polygons with normalized coordinates
    """
    if not isinstance(polygons_raw, list) or len(polygons_raw) == 0:
        return []
    
    shapely_polys = []
    
    for poly in polygons_raw:
        if not isinstance(poly, list) or len(poly) < MIN_POLYGON_POINTS:
            logger.debug(f"Skipping invalid polygon with {len(poly) if isinstance(poly, list) else 0} points")
            continue
        
        # Skip if too many points (likely noise)
        if len(poly) > MAX_POLYGON_POINTS:
            logger.warning(f"Polygon has {len(poly)} points, likely invalid. Skipping.")
            continue
        
        try:
            # Convert normalized [y, x] to pixel [x, y] for Shapely
            pixel_coords = [
                (point[1] * img_width, point[0] * img_height) 
                for point in poly
            ]
            
            # Create Shapely polygon
            shp_poly = Polygon(pixel_coords)
            
            # Validate and fix if needed
            if not shp_poly.is_valid:
                logger.debug("Invalid polygon detected, attempting to fix...")
                shp_poly = validation.make_valid(shp_poly)
            
            # Skip if still invalid or too small
            if not shp_poly.is_valid or shp_poly.is_empty:
                logger.debug("Polygon invalid after repair, skipping")
                continue
            
            if shp_poly.area < MIN_AREA_THRESHOLD:
                logger.debug(f"Polygon area {shp_poly.area:.1f} < {MIN_AREA_THRESHOLD}, skipping")
                continue
            
            # Simplify to reduce vertices (smoother, fewer points)
            simplified = shp_poly.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
            
            # Final area check after simplification
            if simplified.area > MIN_AREA_THRESHOLD:
                shapely_polys.append(simplified)
                logger.debug(f"Polygon accepted: {len(list(simplified.exterior.coords))} points, area={simplified.area:.1f}")
        
        except Exception as e:
            logger.warning(f"Error processing polygon: {e}")
            continue
    
    if not shapely_polys:
        logger.info("No valid polygons after refinement")
        return []
    
    # Merge overlapping polygons (handles partial occlusions)
    try:
        merged = unary_union(shapely_polys)
        
        # Handle both single polygon and MultiPolygon results
        if merged.geom_type == 'Polygon':
            final_polys = [merged]
            logger.info(f"Merged into 1 polygon")
        elif merged.geom_type == 'MultiPolygon':
            final_polys = list(merged.geoms)
            logger.info(f"Merged into {len(final_polys)} polygons")
        else:
            final_polys = shapely_polys
            logger.info(f"No merging needed: {len(final_polys)} polygons")
    except Exception as e:
        logger.warning(f"Polygon merging failed: {e}. Using unmerged polygons.")
        final_polys = shapely_polys
    
    # Convert back to normalized [y, x] coordinates for LabelMe format
    refined = []
    for poly in final_polys:
        # Get exterior coordinates (remove duplicate last point)
        coords = list(poly.exterior.coords)[:-1]
        
        # Convert pixel [x, y] back to normalized [y, x]
        normalized = [
            [y / img_height, x / img_width]
            for x, y in coords
        ]
        
        refined.append(normalized)
    
    logger.info(f"Refinement complete: {len(refined)} final polygons")
    return refined


def validate_polygons(polygons):
    """Basic validation before refinement."""
    if not isinstance(polygons, list):
        logger.warning(f"Expected list for polygons, got {type(polygons)}")
        return []
    
    valid = []
    for poly in polygons:
        if isinstance(poly, list) and len(poly) >= MIN_POLYGON_POINTS:
            # Check all points are valid [y, x] pairs
            if all(isinstance(p, list) and len(p) == 2 for p in poly):
                valid.append(poly)
    
    return valid


# ============================================================================
# IMAGE PROCESSING FUNCTIONS
# ============================================================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def process_single_image(img_path):
    """
    Process a single image with retries.
    
    Returns:
        tuple: (success: bool, message: str, polygons: list)
    """
    try:
        # Get image dimensions for refinement
        with Image.open(img_path) as im:
            img_width, img_height = im.size
        
        # Create model instance (Gemini 2.0 Flash for speed)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,  # Low temperature for consistency
            }
        )
        
        # Upload and process image
        sample_file = genai.upload_file(path=img_path)
        response = model.generate_content([PROMPT, sample_file])
        
        # Clean up uploaded file
        try:
            genai.delete_file(sample_file.name)
        except:
            pass
        
        # Parse response
        result = json.loads(response.text)
        has_fence = result.get("has_fence", False)
        
        if not has_fence:
            logger.info(f"No fence detected: {os.path.basename(img_path)}")
            return False, "no_fence", []
        
        # Validate and refine polygons
        raw_polygons = validate_polygons(result.get("polygons", []))
        
        if not raw_polygons:
            logger.warning(f"No valid polygons returned: {os.path.basename(img_path)}")
            return False, "invalid_polygons", []
        
        # Apply advanced refinement
        refined_polygons = refine_polygons_advanced(raw_polygons, img_width, img_height)
        
        if not refined_polygons:
            logger.warning(f"All polygons filtered out after refinement: {os.path.basename(img_path)}")
            return False, "filtered_out", []
        
        logger.info(f"✓ Processed: {os.path.basename(img_path)} - {len(refined_polygons)} polygons")
        return True, "success", refined_polygons
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error for {os.path.basename(img_path)}: {e}")
        return False, "json_error", []
    
    except Exception as e:
        logger.error(f"Processing error for {os.path.basename(img_path)}: {e}")
        raise  # Will trigger retry


def create_labelme_json(img_path, polygons, img_width, img_height):
    """
    Create LabelMe-compatible JSON annotation.
    
    Args:
        img_path: Path to image file
        polygons: List of polygons with normalized [y, x] coordinates
        img_width: Image width in pixels
        img_height: Image height in pixels
    
    Returns:
        dict: LabelMe JSON structure
    """
    shapes = []
    
    for poly in polygons:
        # Convert normalized [y, x] to pixel [x, y] for LabelMe
        points = [
            [point[1] * img_width, point[0] * img_height]
            for point in poly
        ]
        
        shapes.append({
            "label": "fence",
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        })
    
    return {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(img_path),
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Create output directory for non-fence images
    os.makedirs(NO_FENCE_DIR, exist_ok=True)
    
    # Get all images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    
    # Filter out images that already have JSON annotations
    images_to_process = [
        img for img in image_files
        if not os.path.exists(os.path.splitext(img)[0] + '.json')
    ]
    
    if not images_to_process:
        logger.info("All images already have annotations!")
        return
    
    logger.info(f"Found {len(images_to_process)} images to process")
    logger.info(f"Using {MAX_WORKERS} worker threads")
    logger.info("=" * 70)
    
    # Statistics
    stats = {
        'success': 0,
        'no_fence': 0,
        'error': 0,
        'filtered': 0
    }
    
    # Process with thread pool
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_img = {
            executor.submit(process_single_image, img): img 
            for img in images_to_process
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(images_to_process), desc="Processing images") as pbar:
            for future in as_completed(future_to_img):
                img_path = future_to_img[future]
                
                try:
                    success, message, polygons = future.result()
                    
                    if success and polygons:
                        # Get image dimensions
                        with Image.open(img_path) as im:
                            img_width, img_height = im.size
                        
                        # Create and save JSON
                        json_data = create_labelme_json(img_path, polygons, img_width, img_height)
                        json_path = os.path.splitext(img_path)[0] + '.json'
                        
                        with open(json_path, 'w') as f:
                            json.dump(json_data, f, indent=2)
                        
                        stats['success'] += 1
                    
                    elif message == "no_fence":
                        # Move to no_fence directory
                        dest = os.path.join(NO_FENCE_DIR, os.path.basename(img_path))
                        shutil.move(img_path, dest)
                        stats['no_fence'] += 1
                    
                    elif message == "filtered_out":
                        stats['filtered'] += 1
                    
                    else:
                        stats['error'] += 1
                
                except Exception as e:
                    logger.error(f"Failed to process {os.path.basename(img_path)}: {e}")
                    stats['error'] += 1
                
                pbar.update(1)
    
    # Print summary
    logger.info("=" * 70)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"✓ Successfully labeled: {stats['success']}")
    logger.info(f"↗ No fence detected: {stats['no_fence']}")
    logger.info(f"⚠ Filtered out: {stats['filtered']}")
    logger.info(f"✗ Errors: {stats['error']}")
    logger.info(f"Total processed: {sum(stats.values())}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
