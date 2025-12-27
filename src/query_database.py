"""
Database Query Tool - Explore your extraction results
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from database_4 import IndicatorDatabase, quick_query


def main():
    """Demo database queries"""
    
    db = IndicatorDatabase()
    
    print("\n" + "="*70)
    print("📊 SUSTAINABILITY INDICATORS DATABASE EXPLORER")
    print("="*70)
    
    # 1. Overall statistics
    print("\n1️⃣  OVERALL STATISTICS")
    print("-" * 70)
    stats = db.get_statistics()
    print(f"Total Indicators: {stats['total']}")
    print(f"Successfully Extracted: {stats['successful']}")
    print(f"Accuracy Rate: {stats['accuracy']:.1f}%")
    print(f"Average Confidence: {stats['avg_confidence']:.3f}")
    
    # 2. By company
    print("\n2️⃣  BY COMPANY")
    print("-" * 70)
    for comp in stats['by_company']:
        print(f"{comp['company']:10} {comp['found']}/{comp['total']} "
              f"({comp['accuracy']:.1f}%) | Conf: {comp['avg_confidence']:.2f}")
    
    # 3. Scope 1 emissions across companies
    print("\n3️⃣  SCOPE 1 GHG EMISSIONS (Indicator #1)")
    print("-" * 70)
    scope1 = db.get_indicator_by_id(1)
    if not scope1.empty:
        print(scope1.to_string(index=False))
    else:
        print("No data found")
    
    # 4. Company-specific indicators
    print("\n4️⃣  COMPANY BREAKDOWN")
    print("-" * 70)
    for company in ['AIB', 'BBVA', 'BPCE']:
        company_data = db.get_company_indicators(company)
        if not company_data.empty:
            found = len(company_data[company_data['value'].notna()])
            total = len(company_data)
            print(f"{company}: {found}/{total} indicators extracted")
    
    # 5. Missing indicators
    print("\n5️⃣  MISSING INDICATORS")
    print("-" * 70)
    missing = db.get_missing_indicators()
    print(f"Total missing: {len(missing)}")
    if len(missing) > 0:
        print("\nSample (first 5):")
        print(missing.head()[['company', 'indicator_name']].to_string(index=False))
    
    # 6. High confidence extractions
    print("\n6️⃣  HIGH CONFIDENCE EXTRACTIONS (>0.85)")
    print("-" * 70)
    high_conf = quick_query("""
        SELECT company, indicator_name, value, unit, confidence
        FROM indicators
        WHERE confidence > 0.85 AND value IS NOT NULL
        ORDER BY confidence DESC
        LIMIT 10
    """)
    if not high_conf.empty:
        print(high_conf.to_string(index=False))
    else:
        print("No high-confidence extractions found")
    
    # 7. Extraction runs history
    print("\n7️⃣  EXTRACTION RUNS HISTORY")
    print("-" * 70)
    runs = quick_query("""
        SELECT run_id, run_date, total_indicators, successful_extractions, 
               accuracy_rate, processing_time_seconds
        FROM extraction_runs
        ORDER BY run_date DESC
        LIMIT 5
    """)
    if not runs.empty:
        print(runs.to_string(index=False))
    else:
        print("No extraction runs logged")
    
    print("\n" + "="*70)
    print("✓ Database exploration complete!")
    print("="*70)


if __name__ == "__main__":
    main()
