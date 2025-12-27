"""
Main Runner - Complete extraction pipeline with database integration
"""
import time
from pathlib import Path
import re
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import modules - use the new naming
try:
    from config_1 import *
except:
    from config_1 import *

try:
    from embedder_2 import PDFEmbedder
except:
    from embedder_2 import PDFEmbedder

try:
    from extractor_3 import DataExtractor
except:
    from extractor_3 import DataExtractor

try:
    from database_4 import IndicatorDatabase
except:
    from database_4 import IndicatorDatabase


def detect_company_from_filename(filename: str) -> str:
    """Detect company name from PDF filename"""
    filename_upper = filename.upper()
    
    if 'AIB' in filename_upper:
        return 'AIB'
    elif 'BBVA' in filename_upper:
        return 'BBVA'
    elif 'BPCE' in filename_upper:
        return 'BPCE'
    else:
        name = re.split(r'[_\s\(\)]', filename)[0]
        return name[:20]


def process_document(pdf_path: Path, embedder: PDFEmbedder, 
                     extractor: DataExtractor, indicators: list, 
                     database: IndicatorDatabase) -> list:
    """Process single PDF document"""
    
    # Detect company
    company = detect_company_from_filename(pdf_path.stem)
    year = 2024
    
    print("\n" + "="*70)
    print(f"📊 PROCESSING: {company} ({pdf_path.name})")
    print("="*70)
    
    # Check if embeddings exist
    embedding_path = Path(EMBEDDINGS_DIR) / f"{pdf_path.stem}.pkl"
    
    if embedding_path.exists():
        print(f"📂 Loading cached embeddings...")
        embedder.load_embeddings(str(embedding_path))
    else:
        # Embed document
        success = embedder.embed_document(str(pdf_path), company, year)
        if not success:
            print("❌ Embedding failed")
            return []
        
        # Save embeddings for reuse
        embedder.save_embeddings(str(embedding_path))
    
    # Extract indicators
    print(f"\n🔍 Extracting {len(indicators)} indicators...\n")
    results = []
    
    for i, indicator in enumerate(indicators, 1):
        print(f"[{i}/{len(indicators)}] {indicator['name']}...", end=' ', flush=True)
        
        # Use improved hybrid retrieval
        retrieved = embedder.retrieve_multi_query_improved(
            queries=indicator['queries'],
            keywords=indicator['keywords'],
            unit=indicator['unit'],
            top_k=TOP_K_RETRIEVAL
        )
        
        if not retrieved:
            print(f"✗ No chunks retrieved")
            result = extractor._empty_result(indicator, company, year, "No relevant chunks")
            results.append(result)
            continue
        
        # Extract with Gemini
        chunks = [chunk for chunk, meta, score in retrieved]
        result = extractor.extract_indicator(chunks, indicator, company, year)
        
        # Print result
        if result.value is not None:
            print(f"✓ {result.value} {result.unit} (conf: {result.confidence:.2f})")
        else:
            print(f"✗ Not found")
        
        results.append(result)
    
    return results


def main():
    """Main execution"""
    
    print("\n" + "="*70)
    print("🌱 SUSTAINABILITY INDICATOR EXTRACTION SYSTEM")
    print("   Hybrid RAG + Gemini API")
    print("="*70)
    
    # Initialize database
    database = IndicatorDatabase(db_path=DB_PATH)
    
    # Add company metadata
    database.add_company("AIB", "Ireland", "https://www.aib.ie")
    database.add_company("BBVA", "Spain", "https://shareholdersandinvestors.bbva.com")
    database.add_company("BPCE", "France", "https://www.groupebpce.com")
    
    # Initialize components
    print("\n📦 Initializing components...")
    embedder = PDFEmbedder(
        model_name=EMBEDDING_MODEL,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    extractor = DataExtractor(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL
    )
    
    # Find PDF files
    reports_dir = Path("reports")
    if not reports_dir.exists():
        reports_dir.mkdir(parents=True)
        print(f"\n⚠ Created 'reports/' directory - put your PDF files there!")
        return
    
    pdf_files = list(reports_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"\n⚠ No PDF files found in 'reports/' directory")
        return
    
    print(f"\n📄 Found {len(pdf_files)} PDF reports:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Process each document
    start_time = time.time()
    all_results = []
    
    for pdf_path in pdf_files:
        results = process_document(pdf_path, embedder, extractor, INDICATORS, database)
        all_results.extend(results)
    
    processing_time = time.time() - start_time
    
    # Convert to dict for storage
    results_dict = [
        {
            'company': r.company,
            'report_year': r.year,
            'indicator_id': r.indicator_id,
            'indicator_name': r.indicator_name,
            'value': r.value,
            'unit': r.unit,
            'confidence': r.confidence,
            'source_page': getattr(r, 'source_page', None),
            'source_section': r.source_section,
            'notes': r.notes,
            'raw_text': r.raw_text,
            'extraction_method': 'Hybrid RAG + Gemini'
        }
        for r in all_results
    ]
    
    # Save to database
    print(f"\n💾 Saving to database...")
    database.insert_batch(results_dict)
    
    # Log extraction run
    total_indicators = len(all_results)
    successful = len([r for r in all_results if r.value is not None])
    
    run_id = database.log_extraction_run(
        total_indicators=total_indicators,
        successful=successful,
        processing_time=processing_time,
        model_used=GEMINI_MODEL,
        notes=f"Processed {len(pdf_files)} reports with hybrid retrieval"
    )
    
    print(f"✓ Database updated (Run ID: {run_id})")
    
    # Export to CSV
    print(f"💾 Exporting to CSV...")
    database.export_to_csv(CSV_OUTPUT)
    
    # Print statistics
    stats = database.get_statistics()
    
    print(f"\n" + "="*70)
    print(f"✅ EXTRACTION COMPLETE")
    print(f"="*70)
    print(f"⏱  Time: {processing_time:.1f} seconds")
    print(f"📊 Extracted: {stats['successful']}/{stats['total']} ({stats['accuracy']:.1f}%)")
    print(f"📁 CSV: {CSV_OUTPUT}")
    print(f"💾 Database: {DB_PATH}")
    
    print(f"\n📈 By Company:")
    for comp in stats['by_company']:
        print(f"  {comp['company']:10} {comp['found']}/{comp['total']} "
              f"({comp['accuracy']:.1f}%) | Avg confidence: {comp['avg_confidence']:.2f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
