"""
RAG-based Sustainability Indicator Extraction Engine
"""
import os
import json
import re
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import time

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


@dataclass
class ExtractionResult:
    """Result of indicator extraction"""
    indicator_id: int
    indicator_name: str
    value: Optional[float]
    unit: str
    confidence: float
    source_page: Optional[int]
    source_section: str
    notes: str
    raw_text: str = ""


class RAGExtractor:
    """RAG-based extraction using embeddings + Groq LLM"""
    
    def __init__(
        self,
        api_keys: List[str],
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 3
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.llm_model = llm_model
        
        # Initialize LLM clients FIRST (before any return statements)
        self.llm_clients = []
        self.current_key_index = 0
        self.key_stats = []
        
        # Initialize embedding model
        if EMBEDDING_AVAILABLE:
            print(f"📦 Loading embedding model: {embedding_model}...")
            
            # Disable token auth for public models
            import os
            os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
            
            try:
                self.embedder = SentenceTransformer(
                    embedding_model,
                    trust_remote_code=True,
                    token=False
                )
                print(f"✓ Embedding model loaded")
            except Exception as e:
                print(f"⚠ Error loading model: {e}")
                self.embedder = None
        else:
            self.embedder = None
            print("⚠ Install: pip install sentence-transformers faiss-cpu")
        
        # Initialize Groq clients (MOVED AFTER embedding, but still in __init__)
        if GROQ_AVAILABLE:
            for i, api_key in enumerate(api_keys):
                if api_key and not api_key.startswith("gsk_PUT"):
                    try:
                        client = Groq(api_key=api_key)
                        self.llm_clients.append(client)
                        self.key_stats.append({"requests": 0, "rate_limits": 0, "is_active": True})
                        print(f"✓ Groq API Key #{i+1} ready")
                    except Exception as e:
                        print(f"⚠ API Key #{i+1} failed: {e}")
        
        if not self.llm_clients:
            print("⚠ No valid Groq API keys - set them in config.py")
        
        # Vector store
        self.chunks = []
        self.chunk_metadata = []
        self.index = None
        self.document_embedded = False


    def chunk_document(self, text: str) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        text_length = len(text)
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def embed_document(self, text: str, company_name: str = "Unknown", report_year: int = 2024):
        """Embed entire document into vector database"""
        
        if not self.embedder:
            return False
        
        print(f"\n📚 Embedding: {company_name} {report_year}")
        print(f"   Document: {len(text):,} characters")
        
        # Chunk document
        self.chunks = self.chunk_document(text)
        self.chunk_metadata = [
            {"chunk_id": i, "company": company_name, "year": report_year}
            for i in range(len(self.chunks))
        ]
        
        print(f"   Created: {len(self.chunks)} chunks")
        
        # Create embeddings
        print(f"   Embedding chunks...")
        embeddings = self.embedder.encode(
            self.chunks,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        self.document_embedded = True
        print(f"✓ Ready: {self.index.ntotal} chunks indexed\n")
        return True
    
    def retrieve_relevant_chunks(self, query: str) -> List[Tuple[str, Dict, float]]:
        """Retrieve most relevant chunks for a query"""
        
        if not self.document_embedded:
            return []
        
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, self.top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], self.chunk_metadata[idx], float(score)))
        
        return results
    
    def _get_active_client(self):
        """Get current active LLM client"""
        if not self.llm_clients:
            return None, -1
        
        if self.key_stats[self.current_key_index]["is_active"]:
            return self.llm_clients[self.current_key_index], self.current_key_index
        
        for i in range(len(self.llm_clients)):
            if self.key_stats[i]["is_active"]:
                self.current_key_index = i
                return self.llm_clients[i], i
        
        return None, -1
    
    def _switch_to_next_key(self):
        """Switch to next API key"""
        for _ in range(len(self.llm_clients)):
            self.current_key_index = (self.current_key_index + 1) % len(self.llm_clients)
            if self.key_stats[self.current_key_index]["is_active"]:
                print(f"🔄 Switched to API Key #{self.current_key_index + 1}")
                return True
        return False
    
    def extract_indicator(self, indicator: Dict) -> ExtractionResult:
        """Extract single indicator using RAG"""
        
        if not self.document_embedded:
            return ExtractionResult(
                indicator_id=indicator['id'],
                indicator_name=indicator['name'],
                value=None,
                unit=indicator['unit'],
                confidence=0.0,
                source_page=None,
                source_section="Not embedded",
                notes="Call embed_document() first",
                raw_text=""
            )
        
        # Build search query
        search_query = f"{indicator['name']} {indicator['unit']} {' '.join(indicator['keywords'])} {indicator.get('context', '')}"
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(search_query)
        
        if not relevant_chunks:
            return ExtractionResult(
                indicator_id=indicator['id'],
                indicator_name=indicator['name'],
                value=None,
                unit=indicator['unit'],
                confidence=0.0,
                source_page=None,
                source_section="No chunks",
                notes="No relevant content found",
                raw_text=""
            )
        
        # Combine chunks
        combined_text = "\n\n---\n\n".join([chunk for chunk, meta, score in relevant_chunks])
        
        # Extract with LLM
        client, key_index = self._get_active_client()
        
        if client:
            return self._extract_with_llm(combined_text, indicator)
        else:
            return self._extract_with_regex(combined_text, indicator)
    
    def _extract_with_llm(self, text: str, indicator: Dict, retry_count: int = 0) -> ExtractionResult:
        """Extract using LLM"""
        
        client, key_index = self._get_active_client()
        if not client:
            return self._extract_with_regex(text, indicator)
        
        prompt = f"""Extract "{indicator['name']}" from this sustainability report.

INDICATOR: {indicator['name']}
UNIT: {indicator['unit']}
KEYWORDS: {', '.join(indicator['keywords'])}

CONTENT:
{text[:8000]}

Find the most recent 2024 value. Return ONLY JSON:
{{
  "value": <number or null>,
  "confidence": <0.0 to 1.0>,
  "source_section": "<section name>",
  "notes": "<year and context>",
  "raw_text": "<exact text>"
}}
"""
        
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": "Extract data, return only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=600
            )
            
            self.key_stats[key_index]["requests"] += 1
            result_text = response.choices[0].message.content.strip()
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                return ExtractionResult(
                    indicator_id=indicator['id'],
                    indicator_name=indicator['name'],
                    value=data.get("value"),
                    unit=indicator['unit'],
                    confidence=float(data.get("confidence", 0.0)),
                    source_page=None,
                    source_section=data.get("source_section", "Unknown"),
                    notes=data.get("notes", ""),
                    raw_text=data.get("raw_text", "")[:200]
                )
        
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(x in error_msg.lower() for x in ["rate_limit", "429", "quota"])
            
            if is_rate_limit:
                self.key_stats[key_index]["rate_limits"] += 1
                if self._switch_to_next_key() and retry_count < 2:
                    return self._extract_with_llm(text, indicator, retry_count + 1)
                else:
                    time.sleep(60)
                    self.current_key_index = 0
        
        return self._extract_with_regex(text, indicator)
    
    def _extract_with_regex(self, text: str, indicator: Dict) -> ExtractionResult:
        """Regex fallback"""
        text_lower = text.lower()
        
        for keyword in indicator['keywords']:
            if keyword.lower() in text_lower:
                pos = text_lower.find(keyword.lower())
                context = text[max(0, pos-100):min(len(text), pos+400)]
                
                numbers = re.findall(r'([0-9,]+(?:\.[0-9]+)?)', context)
                if numbers:
                    try:
                        value = float(numbers[0].replace(',', ''))
                        if 0 < value < 3000:
                            return ExtractionResult(
                                indicator_id=indicator['id'],
                                indicator_name=indicator['name'],
                                value=value,
                                unit=indicator['unit'],
                                confidence=0.35,
                                source_page=None,
                                source_section="Regex",
                                notes=f"Found near '{keyword}'",
                                raw_text=context[:150]
                            )
                    except:
                        pass
        
        return ExtractionResult(
            indicator_id=indicator['id'],
            indicator_name=indicator['name'],
            value=None,
            unit=indicator['unit'],
            confidence=0.0,
            source_page=None,
            source_section="Not found",
            notes="Not found",
            raw_text=""
        )
    
    def extract_all_indicators(self, indicators: List[Dict]) -> List[ExtractionResult]:
        """Extract all indicators"""
        
        results = []
        total = len(indicators)
        
        print(f"\n🔍 Extracting {total} indicators...\n")
        
        for i, indicator in enumerate(indicators, 1):
            print(f"[{i}/{total}] {indicator['name']}...", end=' ')
            
            result = self.extract_indicator(indicator)
            
            if result.value is not None:
                print(f"✓ {result.value} {result.unit} (conf: {result.confidence:.2f})")
            else:
                print(f"✗")
            
            results.append(result)
        
        return results
