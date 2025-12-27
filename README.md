

================================================================================
    ESG DATA EXTRACTOR - Automated Sustainability Indicator Extraction
================================================================================


WHAT IT DOES
============

Automatically extracts 20 sustainability metrics from bank PDF reports using AI.

Extracts stuff like:
  - GHG emissions (Scope 1, 2, 3)
  - Energy consumption & renewable %
  - Employee data (headcount, gender %, turnover)
  - Board diversity & meetings
  - Green financing volumes
  - Work accidents, pay gaps, etc.

Technology: RAG (Retrieval-Augmented Generation) + Groq LLM


QUICK START
===========

1. Install dependencies:
   pip install sentence-transformers faiss-cpu groq PyPDF2 pandas

2. Get free API key:
   https://console.groq.com/keys

3. Add your key to src/config.py:
   GROQ_API_KEYS = ["gsk_your_key_here"]

4. Put PDF reports in reports/ folder

5. Run:
   python src/main.py

6. Results saved to: output/extracted_indicators_rag.csv


HOW IT WORKS
============

1. Chunks PDF into ~2,800 pieces
2. Creates vector embeddings (semantic search)
3. Retrieves relevant chunks for each indicator
4. LLM extracts exact values with validation
5. Outputs structured CSV with confidence scores


RESULTS
=======

Tested on 3 major European banks (AIB, BBVA, BPCE):

  Overall Accuracy: 78% (47/60 indicators found)
  
  Company breakdown:
    - AIB:  16/20 (80%) ✓
    - BBVA: 18/20 (90%) ✓✓
    - BPCE: 13/20 (65%)

  Processing speed: ~2-3 minutes per report
  
  Confidence levels:
    - High (0.9-1.0): 65% of found values
    - Medium (0.7-0.9): 25%
    - Low (<0.7): 10%


WHAT YOU GET
============

CSV with:
  - Company name
  - Indicator name & value
  - Unit (tCO₂e, %, days, etc.)
  - Confidence score (0.0 to 1.0)
  - Source section in report
  - Extraction notes

Example output:
  AIB,2024,Total Scope 1 GHG Emissions,2858.0,tCO₂e,0.95
  BBVA,2024,Gender Pay Gap,28.3,%,1.00
  BPCE,2024,Board Meetings,6.0,count/year,0.90


TROUBLESHOOTING
===============

"No valid API keys" → Add real Groq key to config.py
"No PDFs found" → Put .pdf files in reports/ folder
HuggingFace errors → Run: huggingface-cli logout


PROJECT FILES
=============

src/
  - main.py               # Run this
  - rag_extractor.py      # Extraction engine
  - config.py             # API keys & indicators

reports/                  # Put PDFs here
output/                   # Results go here


TECH STACK
==========

- Python 3.10+
- SentenceTransformers (embeddings)
- FAISS (vector search)
- Groq API (LLM)
- PyPDF2 (PDF parsing)


Cost: FREE (using Groq free tier)


================================================================================
