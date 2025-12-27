"""
SQLite Database Manager for Sustainability Indicators
Stores extraction results with full audit trail
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime


class IndicatorDatabase:
    """SQLite database for storing extracted sustainability indicators"""
    
    def __init__(self, db_path: str = "output/sustainability_indicators.db"):
        """Initialize database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = None
        self.create_tables()
    
    def connect(self):
        """Create database connection"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        return self.conn
    
    def create_tables(self):
        """Create database schema"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Main indicators table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company TEXT NOT NULL,
                report_year INTEGER NOT NULL,
                indicator_id INTEGER NOT NULL,
                indicator_name TEXT NOT NULL,
                value REAL,
                unit TEXT NOT NULL,
                confidence REAL,
                source_page INTEGER,
                source_section TEXT,
                notes TEXT,
                raw_text TEXT,
                extraction_method TEXT,
                extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(company, report_year, indicator_id)
            )
        """)
        
        # Extraction runs table (audit trail)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS extraction_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_indicators INTEGER,
                successful_extractions INTEGER,
                accuracy_rate REAL,
                processing_time_seconds REAL,
                model_used TEXT,
                notes TEXT
            )
        """)
        
        # Company metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS companies (
                company_id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_name TEXT UNIQUE NOT NULL,
                country TEXT,
                sector TEXT DEFAULT 'Banking',
                report_url TEXT,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_company_year 
            ON indicators(company, report_year)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_indicator_id 
            ON indicators(indicator_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_confidence 
            ON indicators(confidence)
        """)
        
        conn.commit()
        conn.close()
        
        print(f"✓ Database initialized: {self.db_path}")
    
    def insert_indicator(
        self,
        company: str,
        report_year: int,
        indicator_id: int,
        indicator_name: str,
        value: Optional[float],
        unit: str,
        confidence: float,
        source_page: Optional[int] = None,
        source_section: str = "",
        notes: str = "",
        raw_text: str = "",
        extraction_method: str = "RAG"
    ):
        """Insert or update a single indicator"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO indicators (
                    company, report_year, indicator_id, indicator_name,
                    value, unit, confidence, source_page, source_section,
                    notes, raw_text, extraction_method
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                company, report_year, indicator_id, indicator_name,
                value, unit, confidence, source_page, source_section,
                notes, raw_text, extraction_method
            ))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error inserting indicator: {e}")
            return False
        finally:
            conn.close()
    
    def insert_batch(self, results: List[dict]):
        """Insert multiple indicators at once"""
        conn = self.connect()
        cursor = conn.cursor()
        
        inserted = 0
        for result in results:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO indicators (
                        company, report_year, indicator_id, indicator_name,
                        value, unit, confidence, source_page, source_section,
                        notes, raw_text, extraction_method
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.get('company'),
                    result.get('report_year'),
                    result.get('indicator_id'),
                    result.get('indicator_name'),
                    result.get('value'),
                    result.get('unit'),
                    result.get('confidence'),
                    result.get('source_page'),
                    result.get('source_section', ''),
                    result.get('notes', ''),
                    result.get('raw_text', ''),
                    result.get('extraction_method', 'RAG')
                ))
                inserted += 1
            except Exception as e:
                print(f"Error inserting {result.get('indicator_name')}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"✓ Inserted {inserted}/{len(results)} indicators into database")
        return inserted
    
    def log_extraction_run(
        self,
        total_indicators: int,
        successful: int,
        processing_time: float,
        model_used: str = "llama-3.3-70b-versatile",
        notes: str = ""
    ):
        """Log an extraction run for audit trail"""
        conn = self.connect()
        cursor = conn.cursor()
        
        accuracy = (successful / total_indicators * 100) if total_indicators > 0 else 0
        
        cursor.execute("""
            INSERT INTO extraction_runs (
                total_indicators, successful_extractions, accuracy_rate,
                processing_time_seconds, model_used, notes
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (total_indicators, successful, accuracy, processing_time, model_used, notes))
        
        run_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return run_id
    
    def add_company(self, company_name: str, country: str = "", report_url: str = ""):
        """Add company metadata"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO companies (company_name, country, report_url)
                VALUES (?, ?, ?)
            """, (company_name, country, report_url))
            conn.commit()
            return True
        except Exception as e:
            print(f"Error adding company: {e}")
            return False
        finally:
            conn.close()
    
    def get_company_indicators(self, company: str, year: int = 2024) -> pd.DataFrame:
        """Get all indicators for a company and year"""
        conn = self.connect()
        
        query = """
            SELECT * FROM indicators
            WHERE company = ? AND report_year = ?
            ORDER BY indicator_id
        """
        
        df = pd.read_sql_query(query, conn, params=(company, year))
        conn.close()
        
        return df
    
    def get_indicator_by_id(self, indicator_id: int, year: int = 2024) -> pd.DataFrame:
        """Get specific indicator across all companies"""
        conn = self.connect()
        
        query = """
            SELECT company, value, unit, confidence, source_section
            FROM indicators
            WHERE indicator_id = ? AND report_year = ?
            ORDER BY company
        """
        
        df = pd.read_sql_query(query, conn, params=(indicator_id, year))
        conn.close()
        
        return df
    
    def get_extraction_statistics(self) -> dict:
        """Get overall extraction statistics"""
        conn = self.connect()
        cursor = conn.cursor()
        
        # Total indicators
        cursor.execute("SELECT COUNT(*) FROM indicators")
        total = cursor.fetchone()[0]
        
        # Successful extractions (value not null)
        cursor.execute("SELECT COUNT(*) FROM indicators WHERE value IS NOT NULL")
        successful = cursor.fetchone()[0]
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM indicators WHERE value IS NOT NULL")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # By company
        cursor.execute("""
            SELECT company, 
                   COUNT(*) as total,
                   SUM(CASE WHEN value IS NOT NULL THEN 1 ELSE 0 END) as found
            FROM indicators
            GROUP BY company
        """)
        by_company = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_indicators': total,
            'successful_extractions': successful,
            'accuracy_rate': (successful / total * 100) if total > 0 else 0,
            'average_confidence': avg_confidence,
            'by_company': [
                {'company': row[0], 'total': row[1], 'found': row[2], 
                 'rate': row[2]/row[1]*100}
                for row in by_company
            ]
        }
    
    def export_to_csv(self, output_path: str = "output/indicators_from_db.csv"):
        """Export all indicators to CSV"""
        conn = self.connect()
        
        query = """
            SELECT company, report_year, indicator_id, indicator_name,
                   value, unit, confidence, source_section, notes
            FROM indicators
            ORDER BY company, indicator_id
        """
        
        df = pd.read_sql_query(query, conn)
        df.to_csv(output_path, index=False)
        conn.close()
        
        print(f"✓ Exported to CSV: {output_path}")
        return output_path
    
    def get_missing_indicators(self) -> pd.DataFrame:
        """Get list of indicators not found"""
        conn = self.connect()
        
        query = """
            SELECT company, indicator_id, indicator_name, notes
            FROM indicators
            WHERE value IS NULL
            ORDER BY company, indicator_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()


# Utility function for quick queries
def quick_query(query: str, db_path: str = "output/sustainability_indicators.db"):
    """Run a quick SQL query and return results"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df
