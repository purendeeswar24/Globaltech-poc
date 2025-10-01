import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables with defaults
OUT_DIR = os.getenv('OUTPUT_DIR', 'output')
IMAGE_DIR = os.path.join(OUT_DIR, "images")
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '256'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')

# Create output directories
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def is_meaningful_image(img_obj, page_width, page_height, page=None, page_num=None):
    """Check if an image is likely to be an educational diagram."""
    # Common patterns to exclude (case-insensitive)
    EXCLUDE_KEYWORDS = [
        'watermark', 'disha', 'sample', 'copyright', '©', 'preview',
        'ncert', 'neet', 'ug', 'class', 'test', 'demo', 'page', 'pg',
        'confidential', 'do not copy', 'for review', 'draft', 'example',
        'logo', 'icon', 'publisher', 'author', 'illustration', 'cover', 'back cover'
    ]
    
    # Biology/education related keywords that indicate relevant content
    EDUCATION_KEYWORDS = [
        'fig', 'figure', 'diagram', 'chart', 'table', 'structure', 'process',
        'cell', 'organism', 'biology', 'anatomy', 'physiology', 'molecule',
        'system', 'cycle', 'function', 'mechanism', 'process', 'pathway',
        'reaction', 'structure', 'label', 'cross-section', 'microscopic',
        'organism', 'plant', 'animal', 'bacteria', 'virus', 'fungi', 'algae',
        'photosynthesis', 'respiration', 'digestion', 'circulation', 'neural',
        'genetic', 'dna', 'rna', 'protein', 'enzyme', 'metabolism', 'ecosystem'
    ]
    
    # Skip first few pages (covers, prefaces, etc.)
    if page_num is not None and page_num <= 3:  # Skip first 3 pages
        return False
    
    # Get image dimensions and position
    try:
        x0 = max(0, float(img_obj.get('x0', 0)))
        y0 = max(0, float(img_obj.get('top', 0)))
        x1 = min(page_width, float(img_obj.get('x1', 0)))
        y1 = min(page_height, float(img_obj.get('bottom', 0)))
    except (ValueError, TypeError):
        return False
    
    # Calculate dimensions and area
    width = x1 - x0
    height = y1 - y0
    area = width * height
    page_area = page_width * page_height
    
    # 1. Size-based filtering (more strict)
    min_area = 8000  # ~1.1 in² at 72 DPI
    if area < min_area or area > page_area * 0.5:
        return False
    
    # 2. Aspect ratio filtering (allow slightly wider range for diagrams)
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio > 5 or aspect_ratio < 0.2:
        return False
    
    # 3. Position-based filtering (stricter margins)
    is_in_margin = (
        x0 < page_width * 0.03 or
        x1 > page_width * 0.97 or
        y0 < page_height * 0.03 or
        y1 > page_height * 0.97
    )
    
    # 4. Check for header/footer content
    is_in_header = y0 < page_height * 0.08
    is_in_footer = y1 > page_height * 0.92
    
    if (is_in_header or is_in_footer) and area < page_area * 0.25:
        return False
    
    # 5. Analyze surrounding text for educational content
    if page is not None:
        try:
            # Check larger area around the image for context
            margin = min(50, min(width, height) * 0.5)  # 50% of smaller dimension, max 50pt
            
            # Check area above the image (where captions usually are)
            text_above = page.crop((0, max(0, y0 - margin*2), page_width, y0))
            text_above = (text_above.extract_text() or "").lower()
            
            # Check area below the image
            text_below = page.crop((0, y1, page_width, min(page_height, y1 + margin)))
            text_below = (text_below.extract_text() or "").lower()
            
            # Combine text from both areas
            surrounding_text = f"{text_above} {text_below}"
            
            # Skip if there's exclude-worthy text nearby
            if any(keyword in surrounding_text for keyword in EXCLUDE_KEYWORDS):
                return False
            
            # Look for educational content indicators
            has_edu_content = any(keyword in surrounding_text for keyword in EDUCATION_KEYWORDS)
            has_caption = any(word in surrounding_text for word in ['fig', 'figure', 'diagram', 'table', 'chart'])
            
            # If we found educational content or a caption, it's likely a meaningful image
            if has_edu_content or has_caption:
                return True
                
            # If no text at all, be more cautious
            if not surrounding_text.strip() and area < page_area * 0.15:
                return False
                
        except Exception as e:
            print(f"Warning analyzing image context: {str(e)}")
    
    # 6. If we're not in margins and have reasonable size, might be educational
    if not is_in_margin and area > page_area * 0.1:  # At least 10% of page area
        return True
    
    # Default: skip if we're not sure
    return False

def extract_images(page, page_num):
    """Extract and save educational diagrams and figures from a PDF page."""
    images = []
    if not hasattr(page, 'images') or not page.images:
        return images
        
    page_width = page.width
    page_height = page.height
    
    # First pass: collect and pre-filter potential images
    potential_images = []
    for img_idx, img_obj in enumerate(page.images):
        try:
            # Get image coordinates with boundary checking
            x0 = max(0, float(img_obj.get('x0', 0)))
            y0 = max(0, float(img_obj.get('top', 0)))
            x1 = min(page_width, float(img_obj.get('x1', 0)))
            y1 = min(page_height, float(img_obj.get('bottom', 0)))
            
            # Skip if coordinates are invalid
            if x0 >= x1 or y0 >= y1:
                continue
                
            # Skip very small images immediately (better performance)
            if (x1 - x0) * (y1 - y0) < 2000:  # Less than ~0.3 in²
                continue
                
            # Check if this looks like an educational image
            if is_meaningful_image(img_obj, page_width, page_height, page, page_num):
                potential_images.append((img_idx, x0, y0, x1, y1, img_obj))
                
        except Exception as e:
            if 'Cannot set gray stroke color' not in str(e):
                print(f"Warning processing image {img_idx} on page {page_num}: {str(e)}")
            continue
    
    # Sort potential images by area (largest first) and position (top to bottom, left to right)
    potential_images.sort(key=lambda x: (
        -((x[3]-x[1]) * (x[4]-x[2])),  # Area (negative for descending)
        x[2],  # y0 (top position)
        x[1]   # x0 (left position)
    ))
    
    # Process and save the most promising images (limit to 2 per page for quality)
    saved_count = 0
    for img_idx, x0, y0, x1, y1, img_obj in potential_images:
        if saved_count >= 2:  # Limit to 2 best images per page
            break
            
        try:
            # Create image path with page number and index
            img_path = os.path.join(IMAGE_DIR, f"page_{page_num:03d}_img_{img_idx:03d}.png")
            
            # Add a small margin (0.1 inch) to the crop box
            margin = 7.2  # 0.1 inch in points (72 points = 1 inch)
            x0 = max(0, x0 - margin)
            y0 = max(0, y0 - margin)
            x1 = min(page_width, x1 + margin)
            y1 = min(page_height, y1 + margin)
            
            # Calculate final dimensions
            width = x1 - x0
            height = y1 - y0
            
            # Skip if the area is too small after margin adjustment
            if width * height < 5000:  # Less than ~1 in²
                continue
            
            # Adjust resolution based on content size
            resolution = 150  # Default DPI
            if width > 300 or height > 300:  # For larger diagrams
                resolution = 200
            
            # Crop and save the image
            cropped = page.crop((x0, y0, x1, y1))
            im = cropped.to_image(resolution=resolution)
            
            # Save as PNG with optimization
            im.save(img_path, 'PNG', optimize=True, quality=90)
            
            # Verify the saved image
            if os.path.exists(img_path) and os.path.getsize(img_path) > 5000:  # At least 5KB
                try:
                    # Additional check using PIL to ensure it's a valid image
                    with Image.open(img_path) as img_check:
                        img_check.verify()
                    
                    images.append(img_path)
                    saved_count += 1
                    
                    # If we've saved a good image, skip nearby potential duplicates
                    if saved_count > 0:
                        # Remove any potential images that overlap significantly with this one
                        potential_images = [
                            img for img in potential_images
                            if not (
                                img[1] < x1 and img[3] > x0 and  # x-overlap
                                img[2] < y1 and img[4] > y0 and  # y-overlap
                                img[0] != img_idx  # Not the same image
                            )
                        ]
                        
                except Exception as e:
                    # If image verification fails, remove the file
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    print(f"Warning: Invalid image {img_path}: {str(e)}")
            
        except Exception as e:
            if 'tile cannot extend outside image' not in str(e):
                print(f"Warning: Could not save image {img_idx} on page {page_num}: {str(e)}")
            continue
    
    return images

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks."""
    if not text or not text.strip():
        return []
    
    # Clean up the text
    text = ' '.join(text.split())
    words = text.split()
    chunks = []
    
    # Create chunks with word boundaries
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def process_pdf(pdf_path=None):
    """Process PDF and extract educational content with relevant images."""
    # Get PDF path from environment variable if not provided
    if pdf_path is None:
        pdf_path = os.getenv('PDF_PATH')
    
    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
    
    print(f"Processing PDF: {os.path.basename(pdf_path)}")
    
    # Clear previous images
    print("Clearing previous images...")
    for f in os.listdir(IMAGE_DIR):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            try:
                os.remove(os.path.join(IMAGE_DIR, f))
            except Exception as e:
                print(f"Warning: Could not delete old image {f}: {str(e)}")
    
    docs = []
    total_images_extracted = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages to process: {total_pages}")
            
            # Process each page with progress bar
            for pnum, page in enumerate(tqdm(pdf.pages, desc="Processing pages"), 1):
                try:
                    # Extract text (with improved layout preservation)
                    page_text = page.extract_text(
                        x_tolerance=3,  # Slightly more tolerant of small x differences
                        y_tolerance=3,  # Slightly more tolerant of small y differences
                        keep_blank_chars=False,
                        use_text_flow=True,
                        extra_attrs=["fontname", "size"]
                    ) or ""
                    
                    # Only extract images if the page contains educational content
                    has_educational_content = any(
                        keyword in page_text.lower() 
                        for keyword in ['fig', 'diagram', 'chart', 'table', 'structure', 'process']
                    )
                    
                    # Always extract from first few pages to get any important diagrams
                    if has_educational_content or pnum <= 5:
                        image_paths = extract_images(page, pnum)
                        total_images_extracted += len(image_paths)
                    else:
                        image_paths = []
                    
                    # Only process pages with actual content
                    if page_text.strip() or image_paths:
                        # Split text into chunks, preserving paragraphs where possible
                        chunks = chunk_text(
                            page_text,
                            chunk_size=int(os.getenv('CHUNK_SIZE', '1000')),
                            overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
                        )
                        
                        # If no chunks but we have images, create a minimal chunk
                        if not chunks and image_paths:
                            chunks = ["[Diagram/Figure content]"]
                        
                        # Add each chunk to docs with metadata
                        for chunk in chunks:
                            docs.append({
                                "text": chunk,
                                "page": pnum,
                                "images": image_paths if chunk == chunks[-1] else [],  # Only attach images to last chunk of page
                                "has_images": len(image_paths) > 0
                            })
                    
                except Exception as e:
                    print(f"\nWarning: Error processing page {pnum}: {str(e)}")
                    continue
                    
        print(f"\n✅ Extraction complete. Processed {len(docs)} text chunks and extracted {total_images_extracted} images.")
        return docs
                    
    except Exception as e:
        print(f"\n❌ Error opening PDF: {str(e)}")
        raise
        
    return docs

def create_search_index(docs):
    """Create a search index from the documents."""
    print("\nCreating search index...")
    
    # Extract text for embedding
    texts = [doc["text"] for doc in docs]
    
    # Generate embeddings
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    # Create output directory if it doesn't exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # Save index and metadata
    faiss.write_index(index, os.path.join(OUT_DIR, "index.faiss"))
    
    # Prepare metadata for saving
    metadata = []
    for i, doc in enumerate(docs):
        metadata.append({
            "id": i,
            "text": doc["text"],
            "page": doc["page"],
            "images": doc.get("images", []),
            "has_images": doc.get("has_images", False)
        })
    
    # Save metadata
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n Successfully indexed {len(docs)} document chunks")
    print(f"Index and metadata saved to: {os.path.abspath(OUT_DIR)}")
    
    return index, metadata

def generate_structured_response(query, text_results, min_score=1.5):
    """Generate a structured response from search results."""
    # Filter out low-quality matches
    relevant_results = [r for r in text_results if r.get('score', 2.0) < min_score]
    
    if not relevant_results:
        return {
            "answer": "I couldn't find a clear answer to your question in the document.",
            "sources": [],
            "images": []
        }
    
    # Extract and format the best answer
    best_match = relevant_results[0]
    answer = best_match['text']
    
    # If the answer is too short, try to get more context
    if len(answer) < 100 and len(relevant_results) > 1:
        answer += "\n\n" + relevant_results[1]['text']
    
    # Collect all unique images from relevant results
    all_images = []
    for result in relevant_results[:3]:  # Limit to top 3 results for images
        all_images.extend(result.get('images', []))
    
    # Remove duplicate images while preserving order
    seen = set()
    unique_images = [img for img in all_images if not (img in seen or seen.add(img))][:3]  # Max 3 images
    
    # Format sources (page numbers)
    sources = sorted(list(set([r.get('page', '?') for r in relevant_results])))
    
    return {
        "answer": answer.strip(),
        "sources": sources,
        "images": unique_images
    }

def search_similar(query, index, metadata, model, k=5):
    """Search for similar documents to the query with enhanced results."""
    # Encode the query
    query_embedding = model.encode([query])
    
    # Increase k to get more results for better context
    k = min(k * 2, 10)  # Get more results but cap at 10
    D, I = index.search(query_embedding.astype('float32'), k)
    
    # Get the results with additional context
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < len(metadata):
            result = metadata[idx].copy()
            result['score'] = float(score)
            
            # Add more context from surrounding text if score is good
            if score < 1.5:  # Only for good matches
                # Try to get the next chunk for more context
                if idx + 1 < len(metadata) and metadata[idx]['page'] == metadata[idx+1]['page']:
                    result['text'] += "\n\n" + metadata[idx+1]['text']
            
            results.append(result)
    
    # Generate a structured response
    response = generate_structured_response(query, results)
    return response

def interactive_search(index, metadata):
    """Run an interactive search session."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("\n" + "="*60)
    print("RAG System Ready!")
    print("Enter your questions about the document.")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\n🔍 Your question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("\n👋 Goodbye!")
                break
                
            if not query:
                continue
                
            # Search for relevant content
            results = search_similar(query, index, metadata, model, k=3)
            
            if not results:
                print("\nNo relevant results found. Try rephrasing your question.")
                continue
                
            print(f"\n📄 Found {len(results)} relevant results:")
            
            for i, result in enumerate(results, 1):
                print(f"\n{'='*60}")
                print(f"📄 Result {i} (Page {result.get('page', 'N/A')}, Score: {result.get('score', 0):.2f})")
                print("-"*60)
                print(result['text'][:500] + ("..." if len(result['text']) > 500 else ""))
                
                # Show images if available
                if result.get('images'):
                    print("\n🖼️  Related images:")
                    for img_path in result['images']:
                        if os.path.exists(img_path):
                            print(f"- {os.path.basename(img_path)}")
                        else:
                            print(f"- Image not found: {os.path.basename(img_path)}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        # Process the PDF
        docs = process_pdf()
        
        # Create search index
        index, metadata = create_search_index(docs)
        
        # Start interactive search
        interactive_search(index, metadata)
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please check if the PDF file exists and is accessible.")
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user.")
# We embed every doc (text or image-caption) using Sentence-Transformers and index with FAISS (free).
                    