import os
import pypdf
import pdfplumber
import docx
from typing import List, Dict, Any

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF file.
    Returns a list of dictionaries with 'text' and 'page' (1-indexed).
    """
    pages_content = []
    
    # Try using pdfplumber for better extraction quality (handles tables & layouts)
    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages_content.append({
                        "text": text,
                        "page": i + 1
                    })
    except Exception as e:
        print(f"pdfplumber failed or not available, falling back to pypdf: {e}")
        # Fallback to pypdf
        try:
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_content.append({
                            "text": text,
                            "page": i + 1
                        })
        except Exception as fallback_err:
            print(f"pypdf fallback also failed: {fallback_err}")
            raise Exception(f"Failed to parse PDF: {fallback_err}")
            
    return pages_content

def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from a Word document (.docx).
    Word doesn't have standard page numbers easily extractable in plain text,
    so we group text by paragraphs and return them with a simulated page or section count.
    """
    doc = docx.Document(file_path)
    full_text = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text)
            
    # Also extract text from tables
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                full_text.append(" | ".join(row_text))
                
    joined_text = "\n\n".join(full_text)
    
    # Return as a single section/page
    return [{"text": joined_text, "page": 1}]

def extract_text_from_text_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from TXT or Markdown files.
    Tries UTF-8 decoding, with fallback to CP1252 (Windows standard) or ignoring errors.
    """
    encodings = ["utf-8", "latin-1", "cp1252", "utf-16"]
    text = ""
    
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
            
    if not text:
        # Final fallback with error replacement
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
            
    return [{"text": text, "page": 1}]

def parse_document(file_path: str, file_type: str) -> List[Dict[str, Any]]:
    """
    Router to parse a document based on its extension.
    Returns a list of dicts: [{"text": str, "page": int}]
    """
    ext = file_type.lower() or os.path.splitext(file_path)[1].lower()
    
    if ext in [".pdf", "pdf"]:
        return extract_text_from_pdf(file_path)
    elif ext in [".docx", "docx"]:
        return extract_text_from_docx(file_path)
    elif ext in [".txt", ".md", ".markdown", "txt", "md", "markdown"]:
        return extract_text_from_text_file(file_path)
    else:
        # Fallback to standard text parser
        return extract_text_from_text_file(file_path)
