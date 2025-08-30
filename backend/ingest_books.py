#!/usr/bin/env python3
"""
PDF Ingestion Script for Books Folder

This script ingests all PDF files from the Books folder into the Qdrant vector database
using the existing document processing, chunking, and embedding services.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List

# Ensure project root is on sys.path for standalone execution
# File is located at backend/ingest_books.py → project root is one level up
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.app.services.document_processor import document_processor
from backend.app.services.chunking_service import chunking_service
from backend.app.services.qdrant_service import qdrant_service
from backend.app.services.bge_service import bge_service
from shared.models.schemas import ChunkingMethod

class BookIngestionService:
    def __init__(self):
        self.books_folder = PROJECT_ROOT / "Books"
        self.processed_files = []
        self.failed_files = []
        
    async def ingest_all_books(self):
        """Ingest all PDF files from the Books folder"""
        print(f"[INGESTION] Starting ingestion from: {self.books_folder}")
        
        # Check if Books folder exists
        if not self.books_folder.exists():
            print(f"[ERROR] Books folder not found at: {self.books_folder}")
            return
        
        # Get all PDF files
        pdf_files = list(self.books_folder.glob("*.pdf"))
        print(f"[INGESTION] Found {len(pdf_files)} PDF files to process")
        
        if not pdf_files:
            print("[WARNING] No PDF files found in Books folder")
            return
        
        # Check services health
        await self._check_services_health()
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[INGESTION] Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            try:
                await self._process_single_book(pdf_file)
                self.processed_files.append(pdf_file.name)
                print(f"[SUCCESS] Completed processing: {pdf_file.name}")
            except Exception as e:
                print(f"[ERROR] Failed to process {pdf_file.name}: {str(e)}")
                self.failed_files.append((pdf_file.name, str(e)))
        
        # Print summary
        self._print_summary()
    
    async def _check_services_health(self):
        """Check if required services are available"""
        print("[HEALTH] Checking service availability...")
        
        # Check Qdrant
        qdrant_healthy = await qdrant_service.check_health()
        if not qdrant_healthy:
            raise Exception("Qdrant service is not available")
        print("[HEALTH] ✓ Qdrant service is healthy")
        
        # Check BGE-M3
        bge_healthy = await bge_service.check_health()
        if not bge_healthy:
            raise Exception("BGE-M3 service is not available")
        print("[HEALTH] ✓ BGE-M3 service is healthy")
        
        # Get collection info
        collection_info = await qdrant_service.get_collection_info()
        print(f"[HEALTH] ✓ Qdrant collection: {collection_info}")
    
    async def _process_single_book(self, pdf_file: Path):
        """Process a single PDF book"""
        filename = pdf_file.name
        file_path = str(pdf_file)
        
        print(f"[PROCESSING] {filename} - Step 1: Document extraction")
        
        # Step 1: Extract text from PDF using document processor
        initial_chunk = await document_processor.process_document(
            file_path=file_path,
            filename=filename,
            chunking_method=ChunkingMethod.RECURSIVE
        )
        
        if not initial_chunk or not initial_chunk.content.strip():
            raise Exception("No content extracted from document")
        
        print(f"[PROCESSING] {filename} - Step 2: Text chunking (content length: {len(initial_chunk.content)} chars)")
        
        # Step 2: Apply semantic chunking to break down the content
        semantic_chunks = chunking_service.apply_chunking(
            chunk=initial_chunk,
            method=ChunkingMethod.RECURSIVE
        )
        
        if not semantic_chunks:
            raise Exception("No chunks created from document")
        
        print(f"[PROCESSING] {filename} - Step 3: Storing {len(semantic_chunks)} chunks in Qdrant (BGE-M3 embeddings generated automatically)")
        
        # Step 3: Store chunks in Qdrant (BGE-M3 embeddings are generated automatically in add_chunks)
        success = await qdrant_service.add_chunks(semantic_chunks)
        
        if not success:
            raise Exception("Failed to store chunks in Qdrant")
        
        print(f"[PROCESSING] {filename} - Successfully ingested {len(semantic_chunks)} chunks with BGE-M3 embeddings")
    
    def _print_summary(self):
        """Print ingestion summary"""
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"Total files processed: {len(self.processed_files) + len(self.failed_files)}")
        print(f"Successfully processed: {len(self.processed_files)}")
        print(f"Failed: {len(self.failed_files)}")
        
        if self.processed_files:
            print(f"\n✓ Successfully processed files:")
            for filename in self.processed_files:
                print(f"  - {filename}")
        
        if self.failed_files:
            print(f"\n✗ Failed files:")
            for filename, error in self.failed_files:
                print(f"  - {filename}: {error}")
        
        print("="*60)


async def main():
    """Main function to run the ingestion process"""
    try:
        ingestion_service = BookIngestionService()
        await ingestion_service.ingest_all_books()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Ingestion process interrupted by user")
    except Exception as e:
        print(f"\n[FATAL ERROR] Ingestion process failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)