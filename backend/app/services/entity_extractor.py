"""
Kreuzberg Entity Extraction Service (Optional Feature)

This module provides advanced entity extraction and metadata analysis capabilities
using the Kreuzberg library. It's designed as an optional feature that can be used
independently of the main document processing pipeline.

Author: Claude Code Assistant
Created: 2025-07-14
"""

from kreuzberg import extract_file_sync
from kreuzberg.exceptions import KreuzbergError
from typing import Dict, Any, List, Set, Optional
import os
import re
from datetime import datetime
from collections import Counter

# Optional spaCy import for advanced NLP features
try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    STOP_WORDS = set()
    SPACY_AVAILABLE = False


class KreuzbergEntityExtractor:
    """
    Optional service for advanced entity extraction and metadata analysis using Kreuzberg.
    
    This service provides comprehensive document analysis including:
    - Document metadata extraction (title, author, creation date, etc.)
    - Structured data extraction (tables, embedded resources)
    - Basic entity recognition (emails, dates, URLs, phone numbers)
    - Language detection and content analysis
    
    Note: This is NOT integrated into the main processing pipeline.
          It's provided as an optional feature for advanced use cases.
    """
    
    def __init__(self):
        self.supported_entities = ['emails', 'dates', 'urls', 'phone_numbers', 'spacy_entities']
        self.supported_keywords = ['frequency_based', 'spacy_keywords', 'noun_phrases']
        self.nlp = None
        
        # Initialize spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
    
    def extract_comprehensive_data(self, file_path: str) -> Dict[str, Any]:
        """
        Extract entities, metadata, and structured data using Kreuzberg
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - content: Extracted text content
                - metadata: Document metadata (title, author, etc.)
                - language: Detected language
                - structured_data: Tables and embedded resources
                - entities: Extracted entities
                - file_info: Basic file information
                - extraction_timestamp: When the extraction was performed
        """
        try:
            # Use Kreuzberg's synchronous extraction with full metadata
            result = extract_file_sync(file_path)
            
            # Initialize return dictionary
            extraction_result = {
                "content": "",
                "metadata": {},
                "language": "unknown",
                "structured_data": {
                    "tables": [],
                    "embedded_resources": []
                },
                "entities": {},
                "keywords": {},
                "file_info": {
                    "filename": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "file_type": os.path.splitext(file_path)[1].lower(),
                    "absolute_path": os.path.abspath(file_path)
                },
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            # Extract basic content
            if result and hasattr(result, 'content'):
                extraction_result["content"] = result.content or ""
            
            # Extract metadata if available
            extraction_result["metadata"] = self._extract_metadata(result)
            
            # Set language from metadata if available
            if extraction_result["metadata"].get("language"):
                extraction_result["language"] = extraction_result["metadata"]["language"]
            
            # Extract structured data (tables, embedded resources) if available
            extraction_result["structured_data"] = self._extract_structured_data(result)
            
            # Perform entity extraction
            extraction_result["entities"] = self._extract_entities(extraction_result["content"])
            
            # Perform keyword extraction
            extraction_result["keywords"] = self._extract_keywords(extraction_result["content"])
            
            return extraction_result
            
        except KreuzbergError as e:
            raise Exception(f"Kreuzberg extraction failed for {file_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error during Kreuzberg extraction: {str(e)}")
    
    def _extract_metadata(self, result) -> Dict[str, Any]:
        """Extract document metadata from Kreuzberg result"""
        metadata = {}
        
        if hasattr(result, 'metadata') and result.metadata:
            result_metadata = result.metadata
            metadata = {
                "title": getattr(result_metadata, 'title', None),
                "author": getattr(result_metadata, 'author', None),
                "creator": getattr(result_metadata, 'creator', None),
                "creation_date": getattr(result_metadata, 'creation_date', None),
                "modification_date": getattr(result_metadata, 'modification_date', None),
                "subject": getattr(result_metadata, 'subject', None),
                "keywords": getattr(result_metadata, 'keywords', None),
                "language": getattr(result_metadata, 'language', None),
                "page_count": getattr(result_metadata, 'page_count', None),
                "word_count": getattr(result_metadata, 'word_count', None),
                "producer": getattr(result_metadata, 'producer', None),
                "format": getattr(result_metadata, 'format', None)
            }
        
        return metadata
    
    def _extract_structured_data(self, result) -> Dict[str, List]:
        """Extract structured data from Kreuzberg result"""
        structured_data = {
            "tables": [],
            "embedded_resources": []
        }
        
        if hasattr(result, 'structured_data'):
            structured = result.structured_data
            if hasattr(structured, 'tables'):
                structured_data["tables"] = structured.tables
            if hasattr(structured, 'embedded_resources'):
                structured_data["embedded_resources"] = structured.embedded_resources
        
        return structured_data
    
    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """
        Extract entities using both regex patterns and spaCy NLP
        
        Combines basic regex patterns with advanced spaCy entity recognition
        when available for comprehensive entity extraction.
        """
        if not content:
            return {entity_type: [] for entity_type in self.supported_entities}
        
        entities = {}
        
        # Basic regex-based extraction
        entities["emails"] = list(set(re.findall(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
            content
        )))
        
        # Extract dates (multiple patterns)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD
            r'\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4}\b',  # DD Month YYYY
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b'  # Month DD, YYYY
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        entities["dates"] = list(set(dates))
        
        # Extract URLs
        entities["urls"] = list(set(re.findall(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
            content
        )))
        
        # Extract phone numbers (multiple patterns)
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',           # 123-456-7890 or 123.456.7890
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',             # (123) 456-7890
            r'\+\d{1,3}\s?\d{3,4}\s?\d{3,4}\s?\d{4}',   # International format
            r'\b\d{10}\b'                               # 1234567890
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, content))
        entities["phone_numbers"] = list(set(phones))
        
        # Advanced spaCy-based entity extraction
        entities["spacy_entities"] = self._extract_spacy_entities(content)
        
        return entities
    
    def _extract_spacy_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy NLP processing"""
        if not self.nlp or not content:
            return {}
        
        try:
            doc = self.nlp(content)
            spacy_entities = {}
            
            # Group entities by type
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in spacy_entities:
                    spacy_entities[entity_type] = []
                spacy_entities[entity_type].append(ent.text)
            
            # Remove duplicates and return
            return {k: list(set(v)) for k, v in spacy_entities.items()}
            
        except Exception as e:
            print(f"Warning: spaCy entity extraction failed: {e}")
            return {}
    
    def _extract_keywords(self, content: str) -> Dict[str, List[str]]:
        """
        Extract keywords using multiple methods:
        1. Frequency-based keyword extraction
        2. spaCy-based keyword extraction (noun phrases, key terms)
        3. Noun phrase extraction
        """
        if not content:
            return {method: [] for method in self.supported_keywords}
        
        keywords = {}
        
        # Frequency-based keyword extraction
        keywords["frequency_based"] = self._extract_frequency_keywords(content)
        
        # spaCy-based keyword extraction
        keywords["spacy_keywords"] = self._extract_spacy_keywords(content)
        
        # Noun phrase extraction
        keywords["noun_phrases"] = self._extract_noun_phrases(content)
        
        return keywords
    
    def _extract_frequency_keywords(self, content: str, top_n: int = 20) -> List[str]:
        """Extract keywords based on frequency analysis"""
        if not content:
            return []
        
        # Simple tokenization and frequency analysis
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter out common stop words
        stop_words = STOP_WORDS if STOP_WORDS else {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'a', 'an'
        }
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count frequency and return top keywords
        word_freq = Counter(filtered_words)
        return [word for word, count in word_freq.most_common(top_n)]
    
    def _extract_spacy_keywords(self, content: str, top_n: int = 15) -> List[str]:
        """Extract keywords using spaCy's linguistic features"""
        if not self.nlp or not content:
            return []
        
        try:
            doc = self.nlp(content)
            keywords = []
            
            # Extract important tokens based on POS tags and dependencies
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2 and
                    token.is_alpha):
                    keywords.append(token.lemma_.lower())
            
            # Count frequency and return top keywords
            keyword_freq = Counter(keywords)
            return [word for word, count in keyword_freq.most_common(top_n)]
            
        except Exception as e:
            print(f"Warning: spaCy keyword extraction failed: {e}")
            return []
    
    def _extract_noun_phrases(self, content: str, top_n: int = 10) -> List[str]:
        """Extract noun phrases using spaCy"""
        if not self.nlp or not content:
            return []
        
        try:
            doc = self.nlp(content)
            noun_phrases = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if len(phrase) > 3 and not any(word in phrase for word in ['the', 'a', 'an']):
                    noun_phrases.append(phrase)
            
            # Count frequency and return top phrases
            phrase_freq = Counter(noun_phrases)
            return [phrase for phrase, count in phrase_freq.most_common(top_n)]
            
        except Exception as e:
            print(f"Warning: spaCy noun phrase extraction failed: {e}")
            return []
    
    def extract_metadata_only(self, file_path: str) -> Dict[str, Any]:
        """Extract only metadata without full content processing"""
        try:
            result = extract_file_sync(file_path)
            return {
                "metadata": self._extract_metadata(result),
                "file_info": {
                    "filename": os.path.basename(file_path),
                    "file_size": os.path.getsize(file_path),
                    "file_type": os.path.splitext(file_path)[1].lower()
                },
                "extraction_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise Exception(f"Metadata extraction failed: {str(e)}")
    
    def extract_entities_only(self, file_path: str) -> Dict[str, List[str]]:
        """Extract only entities without metadata or structured data"""
        try:
            result = extract_file_sync(file_path)
            content = getattr(result, 'content', '') if result else ''
            return self._extract_entities(content)
        except Exception as e:
            raise Exception(f"Entity extraction failed: {str(e)}")
    
    def extract_keywords_only(self, file_path: str) -> Dict[str, List[str]]:
        """Extract only keywords without metadata or structured data"""
        try:
            result = extract_file_sync(file_path)
            content = getattr(result, 'content', '') if result else ''
            return self._extract_keywords(content)
        except Exception as e:
            raise Exception(f"Keyword extraction failed: {str(e)}")


# Example usage functions (not for production use)
def example_comprehensive_extraction():
    """
    Example: How to use comprehensive entity and keyword extraction
    
    Usage:
        extractor = KreuzbergEntityExtractor()
        result = extractor.extract_comprehensive_data("/path/to/document.pdf")
        
        # Access different types of extracted data
        content = result["content"]
        metadata = result["metadata"]
        language = result["language"]
        tables = result["structured_data"]["tables"]
        entities = result["entities"]
        keywords = result["keywords"]
        
        print(f"Document title: {metadata.get('title', 'Unknown')}")
        print(f"Language: {language}")
        print(f"Number of tables: {len(tables)}")
        
        # Entity extraction results
        print(f"Emails found: {entities['emails']}")
        print(f"Dates found: {entities['dates']}")
        print(f"spaCy entities (PERSON): {entities['spacy_entities'].get('PERSON', [])}")
        print(f"spaCy entities (ORG): {entities['spacy_entities'].get('ORG', [])}")
        
        # Keyword extraction results
        print(f"Frequency-based keywords: {keywords['frequency_based'][:5]}")
        print(f"spaCy keywords: {keywords['spacy_keywords'][:5]}")
        print(f"Noun phrases: {keywords['noun_phrases'][:3]}")
    """
    pass


def example_metadata_only():
    """
    Example: How to extract only metadata
    
    Usage:
        extractor = KreuzbergEntityExtractor()
        metadata_result = extractor.extract_metadata_only("/path/to/document.pdf")
        
        print(f"Title: {metadata_result['metadata'].get('title')}")
        print(f"Author: {metadata_result['metadata'].get('author')}")
        print(f"Creation Date: {metadata_result['metadata'].get('creation_date')}")
    """
    pass


def example_entities_only():
    """
    Example: How to extract only entities
    
    Usage:
        extractor = KreuzbergEntityExtractor()
        entities = extractor.extract_entities_only("/path/to/document.pdf")
        
        print(f"Emails: {entities['emails']}")
        print(f"Phone numbers: {entities['phone_numbers']}")
        print(f"URLs: {entities['urls']}")
        print(f"spaCy PERSON entities: {entities['spacy_entities'].get('PERSON', [])}")
        print(f"spaCy ORG entities: {entities['spacy_entities'].get('ORG', [])}")
    """
    pass


def example_keywords_only():
    """
    Example: How to extract only keywords
    
    Usage:
        extractor = KreuzbergEntityExtractor()
        keywords = extractor.extract_keywords_only("/path/to/document.pdf")
        
        print(f"Top frequency-based keywords: {keywords['frequency_based'][:10]}")
        print(f"Top spaCy keywords: {keywords['spacy_keywords'][:10]}")
        print(f"Top noun phrases: {keywords['noun_phrases'][:5]}")
    """
    pass


# Create a singleton instance for easy import
entity_extractor = KreuzbergEntityExtractor()