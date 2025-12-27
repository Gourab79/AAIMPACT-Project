"""
Main script to run RAG extraction
USAGE: python run_extraction.py
"""
import os
import PyPDF2
import pandas as pd
from pathlib import Path

# Import config and RAG extractor
import config
from rag_extractor import RAGExtractor


def extract_pdf_text(pdf_path):
    """Extract text from PDF"""
    print(f"📄 Reading PDF: {pdf_path}")
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    print(f"✓ Extracted {len(text):,} characters")
    return text


def process_single_report(pdf_path, extractor, indicators):
    """Process one PDF report"""
    
    # Get company name from filename
    filename = Path(pdf_path).stem
    
    # Extract year using regex (find 4-digit number like 2024)
    import re
    year_match = re.search(r'(20\d{2})', filename)
    report_year = int(year_match.group(1)) if year_match else 2024
    
    # Extract company name (try different patterns)
    if 'AIB' in filename.upper():
        company_name = 'AIB'
    elif 'BBVA' in filename.upper():
        company_name = 'BBVA'
    elif 'BPCE' in filename.upper():
        company_name = 'BPCE'
    else:
        # Use first part before underscore or parenthesis
        company_name = re.split(r'[_\(\)]', filename)[0]
    
    print(f"\n{'='*60}")
    print(f"Processing: {company_name} {report_year}")
    print(f"{'='*60}")
    
    # Extract text
    text = extract_pdf_text(pdf_path)
    
    # Embed document
    extractor.embed_document(text, company_name, report_year)
    
    # Extract all indicators
    results = extractor.extract_all_indicators(indicators)
    
    # Add company info to results
    for result in results:
        result.company = company_name
        result.year = report_year
    
    return results



def main():
    """Main execution"""
    
    print("="*60)
    print("🌱 SUSTAINABILITY INDICATOR EXTRACTION (RAG)")
    print("="*60)
    
    # Initialize RAG extractor
    extractor = RAGExtractor(
        api_keys=config.GROQ_API_KEYS,
        embedding_model=config.EMBEDDING_MODEL,
        llm_model=config.LLM_MODEL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        top_k=config.TOP_K_CHUNKS
    )
    
    # Find all PDF reports
    reports_dir = Path("reports")
    pdf_files = list(reports_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"\n⚠ No PDF files found in {reports_dir}/")
        print(f"   Put your PDF reports there and run again.")
        return
    
    print(f"\nFound {len(pdf_files)} PDF reports:")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")
    
    # Process each report
    all_results = []
    for pdf_path in pdf_files:
        results = process_single_report(pdf_path, extractor, config.INDICATORS)
        all_results.extend(results)
    
    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "company": getattr(r, 'company', 'Unknown'),
            "report_year": getattr(r, 'year', 2024),
            "indicator_id": r.indicator_id,
            "indicator_name": r.indicator_name,
            "value": r.value,
            "unit": r.unit,
            "confidence": r.confidence,
            "source_section": r.source_section,
            "notes": r.notes
        }
        for r in all_results
    ])
    
    # Save to CSV
    output_dir = Path("output_2")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "extracted_indicators_rag.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✅ EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"Total indicators extracted: {len(df[df['value'].notna()])}/{len(df)}")
    print(f"\nSuccess rate by company:")
    for company in df['company'].unique():
        company_df = df[df['company'] == company]
        success = len(company_df[company_df['value'].notna()])
        total = len(company_df)
        print(f"  {company}: {success}/{total} ({success/total*100:.1f}%)")


if __name__ == "__main__":
    main()
