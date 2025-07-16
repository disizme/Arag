# Legacy imports - commented out for Kreuzberg integration
# import fitz  # PyMuPDF
# from docx import Document
# from pptx import Presentation

# New Kreuzberg import for unified document processing
from kreuzberg import extract_file
from kreuzberg.exceptions import KreuzbergError

from typing import List, Union
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
    
    async def extract_text_with_kreuzberg(self, file_path: str) -> str:
        """Extract text from any supported document using Kreuzberg library
        
        Kreuzberg provides unified text extraction for multiple document formats:
        - PDFs (with OCR fallback)
        - Office documents (DOCX, PPTX, XLSX)
        - Images (PNG, JPG, TIFF, etc.)
        - Plain text files
        - And many more formats
        
        Returns:
            str: The extracted text content
        """
        try:
            # Use Kreuzberg's async extraction
            result = await extract_file(file_path)
            
            # Extract content
            content = result.content
            
            # Return the entire content as a single string
            return content.strip() if content else ""
                
        except KreuzbergError as e:
            raise Exception(f"Kreuzberg extraction failed for {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during Kreuzberg extraction: {str(e)}")
    
    def create_initial_document_chunk(
        self, 
        file_path: str, 
        filename: str, 
        content: str # List[Tuple[str, int]] ; Individual Page extractions
    ) -> DocumentChunk:
        """Create initial document chunk from extracted text"""
        document_id = hashlib.md5(f"{filename}_{os.path.getmtime(file_path)}".encode()).hexdigest()
        
        chunk = DocumentChunk(
            id=f"{document_id}_0",
            content=content,
            metadata={
                "filename": filename,
                "document_id": document_id,
                "file_size": os.path.getsize(file_path),
                "document_type": self.get_document_type(filename).value,
                "extraction_method": "kreuzberg"
            },
            source_file=filename,
            page_number=1,
            chunk_index=0,
            created_at=datetime.now()
        )
        
        return chunk
    

    async def process_document(
        self, 
        file_path: str, 
        filename: str, 
        chunking_method: Union[str, ChunkingMethod] = ChunkingMethod.RECURSIVE
    ) -> DocumentChunk:
        """Process a document and return semantic chunks (for testing/standalone use)"""
        try:
            # Step 1: Extract text using Kreuzberg
            content = await self.extract_text_with_kreuzberg(file_path)
            
            if not content:
                raise Exception("No content extracted from document")
            
            # Step 2: Create initial document chunk
            chunks = self.create_initial_document_chunk(file_path, filename, content)
            
            return chunks
        except Exception as e:
            raise Exception(f"Failed to process document {filename}: {str(e)}")

# Global instance
document_processor = DocumentProcessor()