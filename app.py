 

import os
from typing import List, Optional, Dict, Any

import numpy as np
import faiss
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


class InvestmentAnalyzer:
    def __init__(self, embedding_model_name: Optional[str] = None):
        model_name = embedding_model_name or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[INFO] Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        self.chunks: List[str] = []
        self.chunk_pages_map: List[int] = []  # page number for each chunk
        self.index: Optional[faiss.IndexFlatL2] = None
        self.gemini_model = None

    # ---------- PDF + OCR (page-based) ----------

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Returns a list of dicts: [{"page": page_number, "text": "..."}]
        """
        pages: List[Dict[str, Any]] = []

         
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    pages.append({"page": i + 1, "text": txt})
        except Exception as e:
            print(f"[ERROR] pdfplumber failed: {e}")

        total_len = sum(len(p["text"].strip()) for p in pages)
        if total_len >= 50:
            print(f"[INFO] pdfplumber extracted {total_len} characters of text.")
            return pages

        print("[WARN] Very little/no text found. Falling back to OCR...")

        # OCR fallback
        pages = []
        try:
            images = convert_from_path(pdf_path)
            for i, img in enumerate(images):
                text = pytesseract.image_to_string(img) or ""
                pages.append({"page": i + 1, "text": text})
        except Exception as e:
            print(f"[ERROR] OCR failed: {e}")

        total_len = sum(len(p["text"].strip()) for p in pages)
        print(f"[INFO] OCR extracted {total_len} characters of text.")
        return pages

    def chunk_pages(self, pages: List[Dict[str, Any]], chunk_size: int = 450, overlap: int = 50):
        """
        Build chunks while preserving page numbers.
        """
        self.chunks = []
        self.chunk_pages_map = []

        for page in pages:
            words = page["text"].split()
            if not words:
                continue

            step = max(1, chunk_size - overlap)
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk:
                    self.chunks.append(chunk)
                    self.chunk_pages_map.append(page["page"])

        print(f"[INFO] Created {len(self.chunks)} chunks from {len(pages)} pages.")

    def create_vector_store(self):
        """
        Create FAISS index from self.chunks
        """
        if not self.chunks:
            raise ValueError("No chunks to index")

        embeddings = self.embedding_model.encode(self.chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings).astype("float32"))
        print(f"[INFO] Vector store created with {len(self.chunks)} chunks (dim={dim})")

    # ---------- Gemini setup ----------

    def setup_llm(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in .env")

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.gemini_model = genai.GenerativeModel(model_name)
        print(f"[INFO] Gemini model configured: {model_name}")

    # ---------- Full pipeline ----------

    def process_pdf(self, pdf_path: str) -> bool:
        print(f"[INFO] Processing PDF: {pdf_path}")
        pages = self.extract_text_from_pdf(pdf_path)
        if not pages:
            print("[ERROR] No pages extracted.")
            return False

        self.chunk_pages(pages)
        if not self.chunks:
            print("[ERROR] No chunks created.")
            return False

        self.create_vector_store()
        return True

    # ---------- QA: answer + pages ----------

    def retrieve_answer(self, question: str) -> Dict[str, Any]:
        if self.index is None or not self.chunks:
            return {
                "answer": "No document indexed.",
                "pages": [],
                "context_chunks": []
            }

        if self.gemini_model is None:
            raise RuntimeError("Gemini model not initialized. Call setup_llm() first.")

        # Retrieve top 3 chunks
        q_embedding = self.embedding_model.encode([question])
        _, idx = self.index.search(np.array(q_embedding).astype("float32"), 3)

        context_chunks: List[str] = []
        used_pages: set[int] = set()

        for i in idx[0]:
            i = int(i)
            if 0 <= i < len(self.chunks):
                context_chunks.append(self.chunks[i])
                used_pages.add(self.chunk_pages_map[i])

        pages_sorted = sorted(list(used_pages))

        context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""
You are an finanace investment analysis expert.

Use ONLY the following context to answer the question.

Context:
{context}

Question:
{question}

Rules:
- If the answer is fully and explicitly present in the context, answer in detailed approx 10-15 sentences.
- If ANY required information is missing, unclear, or not stated, respond EXACTLY:
  "Not available in document".
- Do NOT guess or infer.
- Do NOT repeat the context or the question.
- If you answer, do NOT mention pages in your text; pages will be handled separately.

Now give ONLY the final answer:
"""

        try:
            response = self.gemini_model.generate_content(prompt)
            answer_text = (response.text or "").strip()
        except Exception as e:
            print(f"[ERROR] Gemini call failed: {e}")
            answer_text = f"Error calling Gemini: {e}"

        return {
            "answer": answer_text,
            "pages": pages_sorted,
            "context_chunks": context_chunks
        }
