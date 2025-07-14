# Legacy imports - commented out for Kreuzberg integration
# import fitz  # PyMuPDF
# from docx import Document
# from pptx import Presentation

# New Kreuzberg import for unified document processing
from kreuzberg import extract_file_sync
from kreuzberg.exceptions import KreuzbergError

from typing import List, Dict, Any, Tuple, Union
import os
import hashlib
from datetime import datetime
from shared.models.schemas import DocumentType, DocumentChunk, ChunkingMethod

class DocumentProcessor:
    def __init__(self):
        self.supported_types = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.pptx': DocumentType.PPTX
        }
    
    def get_document_type(self, filename: str) -> DocumentType:
        """Get document type from filename"""
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.supported_types:
            raise ValueError(f"Unsupported file type: {ext}")
        return self.supported_types[ext]
    
    # Legacy PDF extraction method - commented out for Kreuzberg integration
    # def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
    #     """Extract text from PDF with page numbers using PyMuPDF"""
    #     doc = fitz.open(file_path)
    #     pages = []
    #     
    #     for page_num in range(len(doc)):
    #         page = doc[page_num]
    #         text = page.get_text()
    #         if text.strip():
    #             pages.append((text, page_num + 1))
    #     
    #     doc.close()
    #     return pages
    
    # Legacy DOCX extraction method - commented out for Kreuzberg integration
    # def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
    #     """Extract text from DOCX using python-docx"""
    #     doc = Document(file_path)
    #     pages = []
    #     current_page = 1
    #     
    #     for paragraph in doc.paragraphs:
    #         text = paragraph.text.strip()
    #         if text:
    #             pages.append((text, current_page))
    #     
    #     return pages
    
    # Legacy PPTX extraction method - commented out for Kreuzberg integration
    # def extract_text_from_pptx(self, file_path: str) -> List[Tuple[str, int]]:
    #     """Extract text from PPTX using python-pptx"""
    #     prs = Presentation(file_path)
    #     pages = []
    #     
    #     for slide_num, slide in enumerate(prs.slides):
    #         slide_text = []
    #         for shape in slide.shapes:
    #             if hasattr(shape, "text"):
    #                 slide_text.append(shape.text)
    #         
    #         if slide_text:
    #             combined_text = '\n'.join(slide_text)
    #             pages.append((combined_text, slide_num + 1))
    #     
    #     return pages
    
    # Legacy type-specific extraction method - commented out for Kreuzberg integration
    # def extract_text(self, file_path: str, document_type: DocumentType) -> List[Tuple[str, int]]:
    #     """Extract text from document based on type using legacy methods"""
    #     if document_type == DocumentType.PDF:
    #         return self.extract_text_from_pdf(file_path)
    #     elif document_type == DocumentType.DOCX:
    #         return self.extract_text_from_docx(file_path)
    #     elif document_type == DocumentType.PPTX:
    #         return self.extract_text_from_pptx(file_path)
    #     else:
    #         raise ValueError(f"Unsupported document type: {document_type}")
    
    def extract_text_with_kreuzberg(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from any supported document using Kreuzberg library
        
        Kreuzberg provides unified text extraction for multiple document formats:
        - PDFs (with OCR fallback)
        - Office documents (DOCX, PPTX, XLSX)
        - Images (PNG, JPG, TIFF, etc.)
        - Plain text files
        - And many more formats
        
        Returns:
            List[Tuple[str, int]]: List of (text_content, page_number) tuples
        """
        try:
            # Use Kreuzberg's synchronous extraction
            result = extract_file_sync(file_path)
            
            # Extract content and metadata
            content = result.content
            metadata = result.metadata if hasattr(result, 'metadata') else None
            
            # For now, treat the entire document as a single page
            # Future enhancement: implement page-level extraction if Kreuzberg supports it
            if content and content.strip():
                return [(content, 1)]
            else:
                return []
                
        except KreuzbergError as e:
            raise Exception(f"Kreuzberg extraction failed for {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during Kreuzberg extraction: {str(e)}")
    
    def create_document_chunks(
        self, 
        file_path: str, 
        filename: str, 
        pages: List[Tuple[str, int]],
        chunking_method: Union[str, ChunkingMethod] = ChunkingMethod.RECURSIVE
    ) -> List[DocumentChunk]:
        """Create document chunks from extracted text"""
        chunks = []
        document_id = hashlib.md5(f"{filename}_{os.path.getmtime(file_path)}".encode()).hexdigest()
        
        # Handle both string and enum inputs
        if isinstance(chunking_method, str):
            chunking_method_str = chunking_method
        else:
            chunking_method_str = chunking_method.value
        
        for chunk_index, (text, page_num) in enumerate(pages):
            chunk = DocumentChunk(
                id=f"{document_id}_{chunk_index}",
                content=text,
                metadata={
                    "filename": filename,
                    "document_id": document_id,
                    "chunking_method": chunking_method_str,
                    "file_size": os.path.getsize(file_path)
                },
                source_file=filename,
                page_number=page_num,
                chunk_index=chunk_index,
                created_at=datetime.now()
            )
            chunks.append(chunk)
        
        return chunks
    
    async def process_document(
        self, 
        file_path: str, 
        filename: str, 
        chunking_method: Union[str, ChunkingMethod] = ChunkingMethod.RECURSIVE
    ) -> List[DocumentChunk]:
        """Process a document and return chunks using Kreuzberg for text extraction"""
        try:
            # Use Kreuzberg for unified text extraction instead of type-specific methods
            pages = self.extract_text_with_kreuzberg(file_path)
            chunks = self.create_document_chunks(file_path, filename, pages, chunking_method)
            
            return chunks
        except Exception as e:
            raise Exception(f"Failed to process document {filename}: {str(e)}")

document_processor = DocumentProcessor()