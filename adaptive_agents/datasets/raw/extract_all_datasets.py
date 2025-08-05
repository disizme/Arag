#!/usr/bin/env python3
"""
Unified Dataset Extraction Script

This script processes all raw dataset files including AmbigQA parquet files and other formats,
extracting them into a standardized JSON format with fields: question, source, answer, and unique id.

Supported formats:
- JSONL files (ML-QA, squad, trivia with different structures)
- JSON files (QA-DSML with specific structure)  
- Parquet files (AmbigQA, DS-instruct, QA with binary format)

Special handling for AmbigQA files to convert nq_answer arrays to strings.
"""

import json
import pandas as pd
import uuid
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union


class UnifiedDatasetExtractor:
    def __init__(self):
        self.current_dir = Path(__file__).parent
        self.extracted_dir = self.current_dir / "extracted"
        self.processed_files = []
        self.failed_files = []
        
        # Create extracted directory if it doesn't exist
        self.extracted_dir.mkdir(exist_ok=True)
        
    def extract_all_datasets(self, include_ambigqa: bool = True):
        """Extract all dataset files in the raw_datasets folder"""
        print("[UNIFIED EXTRACTION] Starting comprehensive dataset extraction...")
        
        # Get all files except keywords folder and this script
        files_to_process = []
        for file_path in self.current_dir.iterdir():
            if (file_path.is_file() and 
                file_path.name not in ["extract_all_datasets.py", "extract_datasets.py", "extract_ambigqa.py"] and
                not file_path.name.startswith('.')):
                files_to_process.append(file_path)
        
        print(f"[EXTRACTION] Found {len(files_to_process)} files to process")
        
        # Process each file
        for file_path in files_to_process:
            print(f"\n[PROCESSING] {file_path.name}")
            
            # Special handling for AmbigQA files
            if "AmbigQA" in file_path.name and include_ambigqa:
                try:
                    extracted_data = self._extract_ambigqa_file(file_path)
                    if extracted_data:
                        output_file = self._save_extracted_data(file_path, extracted_data)
                        self.processed_files.append((file_path.name, len(extracted_data), output_file))
                        print(f"[SUCCESS] Extracted {len(extracted_data)} entries from {file_path.name}")
                    else:
                        print(f"[WARNING] No data extracted from {file_path.name}")
                except Exception as e:
                    print(f"[ERROR] Failed to process AmbigQA file {file_path.name}: {str(e)}")
                    self.failed_files.append((file_path.name, str(e)))
            else:
                # Standard extraction for other files
                try:
                    extracted_data = self._extract_from_file(file_path)
                    if extracted_data:
                        output_file = self._save_extracted_data(file_path, extracted_data)
                        self.processed_files.append((file_path.name, len(extracted_data), output_file))
                        print(f"[SUCCESS] Extracted {len(extracted_data)} entries from {file_path.name}")
                    else:
                        print(f"[WARNING] No data extracted from {file_path.name}")
                        
                except Exception as e:
                    print(f"[ERROR] Failed to process {file_path.name}: {str(e)}")
                    self.failed_files.append((file_path.name, str(e)))
        
        # Print summary
        self._print_summary()
    
    def _extract_from_file(self, file_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Extract data from a single file based on its format"""
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.jsonl':
            return self._extract_from_jsonl(file_path)
        elif file_extension == '.json':
            return self._extract_from_json(file_path)
        elif file_extension == '.parquet':
            # Check if it's an AmbigQA file (handled separately)
            if "AmbigQA" not in file_path.name:
                return self._extract_from_parquet(file_path)
            else:
                print(f"[SKIP] AmbigQA file {file_path.name} handled by special method")
                return None
        else:
            print(f"[SKIP] Unsupported file format: {file_extension}")
            return None
    
    def _extract_ambigqa_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract data from AmbigQA parquet file with special nq_answer handling"""
        extracted_data = []
        filename = file_path.name
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            print(f"[INFO] Loaded {len(df)} rows from {filename}")
            print(f"[INFO] Columns: {list(df.columns)}")
            
            # Check if required columns exist
            if 'nq_answer' not in df.columns:
                print(f"[ERROR] 'nq_answer' column not found in {filename}")
                return []
            
            # Look for question column
            question_col = None
            for col in df.columns:
                if 'question' in col.lower():
                    question_col = col
                    break
            
            if not question_col:
                print(f"[ERROR] No question column found in {filename}")
                return []
            
            print(f"[INFO] Using question column: {question_col}")
            print(f"[INFO] Sample nq_answer type: {type(df['nq_answer'].iloc[0])}")
            
            # Process each row
            for index, row in df.iterrows():
                try:
                    question = str(row[question_col]) if pd.notna(row[question_col]) else ""
                    nq_answer = row['nq_answer']
                    
                    # Convert nq_answer array to string
                    answer_string = self._convert_answer_to_string(nq_answer)
                    
                    if question.strip() and answer_string.strip():
                        extracted_data.append({
                            "id": str(uuid.uuid4()),
                            "question": question.strip(),
                            "answer": answer_string.strip(),
                            "source": filename
                        })
                    
                except Exception as e:
                    print(f"[WARNING] Error processing row {index}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"[ERROR] Error processing {filename}: {str(e)}")
            return []
        
        return extracted_data
    
    def _convert_answer_to_string(self, nq_answer) -> str:
        """Convert nq_answer array to a string representation without brackets"""
        try:
            # If it's already a string, return as is
            if isinstance(nq_answer, str):
                return nq_answer
            
            # If it's a list/array, extract and join the elements
            if isinstance(nq_answer, (list, tuple)):
                # Filter out empty strings and None values
                valid_answers = []
                for ans in nq_answer:
                    if ans is not None:
                        ans_str = str(ans).strip()
                        if ans_str and ans_str != 'nan':
                            valid_answers.append(ans_str)
                
                if valid_answers:
                    # Join multiple answers with " | " separator (no brackets)
                    return " | ".join(valid_answers)
                else:
                    return "No answer available"
            
            # If it's a pandas Series or numpy array, convert to list first
            if hasattr(nq_answer, 'tolist'):
                return self._convert_answer_to_string(nq_answer.tolist())
            
            # If it's some other type, convert to string but remove brackets if present
            answer_str = str(nq_answer) if nq_answer is not None else "No answer available"
            
            # Remove array brackets if the string representation includes them
            if answer_str.startswith('[') and answer_str.endswith(']'):
                # Remove brackets and split by comma, then rejoin
                inner_content = answer_str[1:-1]  # Remove [ and ]
                # Split by comma and clean up quotes
                parts = [part.strip().strip('"').strip("'") for part in inner_content.split(',')]
                valid_parts = [part for part in parts if part and part != 'nan']
                if valid_parts:
                    return " | ".join(valid_parts)
            
            return answer_str
            
        except Exception as e:
            print(f"[WARNING] Error converting answer to string: {str(e)}")
            return "Error processing answer"
    
    def _extract_from_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract from JSONL files with different structures"""
        extracted_data = []
        filename = file_path.name
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    data = json.loads(line)
                    entry = self._process_jsonl_entry(data, filename, line_num)
                    if entry:
                        extracted_data.append(entry)
                        
                except json.JSONDecodeError as e:
                    print(f"[WARNING] Invalid JSON on line {line_num + 1}: {str(e)}")
                    continue
                except Exception as e:
                    print(f"[WARNING] Error processing line {line_num + 1}: {str(e)}")
                    continue
        
        return extracted_data
    
    def _process_jsonl_entry(self, data: Dict, filename: str, line_num: int) -> Optional[Dict[str, Any]]:
        """Process a single JSONL entry based on its structure"""
        
        # ML-QA format: {"text": "<start_of_turn>user\n...question...<end_of_turn>\n<start_of_turn>model\n...answer...<end_of_turn>"}
        if "text" in data and "<start_of_turn>" in data["text"]:
            return self._extract_from_gemini_format(data, filename)
            
        # Squad format: {"dataset": "squad", "question_text": "...", "answers_objects": [...], "contexts": [...]}
        elif "dataset" in data and "question_text" in data:
            return self._extract_from_squad_format(data, filename)
            
        # Trivia format: {"dataset": "trivia", "question_text": "...", "answers_objects": [...]}
        elif "dataset" in data and data.get("dataset") == "trivia":
            return self._extract_from_trivia_format(data, filename)
            
        else:
            print(f"[WARNING] Unknown JSONL format in {filename} line {line_num + 1}")
            return None
    
    def _extract_from_gemini_format(self, data: Dict, filename: str) -> Optional[Dict[str, Any]]:
        """Extract from Gemini/ML-QA conversation format"""
        try:
            text = data["text"]
            
            # Split by turns
            if "<start_of_turn>user" in text and "<start_of_turn>model" in text:
                # Extract user question
                user_start = text.find("<start_of_turn>user\n") + len("<start_of_turn>user\n")
                user_end = text.find("<end_of_turn>", user_start)
                question = text[user_start:user_end].strip()
                
                # Extract model answer
                model_start = text.find("<start_of_turn>model\n") + len("<start_of_turn>model\n")
                model_end = text.find("<end_of_turn>", model_start)
                answer = text[model_start:model_end].strip()
                
                return {
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "answer": answer,
                    "source": filename
                }
            
            return None
            
        except Exception as e:
            print(f"[WARNING] Error extracting Gemini format: {str(e)}")
            return None
    
    def _extract_from_squad_format(self, data: Dict, filename: str) -> Optional[Dict[str, Any]]:
        """Extract from Squad format"""
        try:
            question = data.get("question_text", "")
            
            # Extract answer from answers_objects
            answer = ""
            if "answers_objects" in data and data["answers_objects"]:
                answer_obj = data["answers_objects"][0]
                if "spans" in answer_obj and answer_obj["spans"]:
                    answer = answer_obj["spans"][0]
            
            # Use existing question_id if available, otherwise generate
            question_id = data.get("question_id", str(uuid.uuid4()))
            
            return {
                "id": question_id,
                "question": question,
                "answer": answer,
                "source": filename
            }
            
        except Exception as e:
            print(f"[WARNING] Error extracting Squad format: {str(e)}")
            return None
    
    def _extract_from_trivia_format(self, data: Dict, filename: str) -> Optional[Dict[str, Any]]:
        """Extract from Trivia format"""
        try:
            question = data.get("question_text", "")
            
            # Extract answer from answers_objects
            answer = ""
            if "answers_objects" in data and data["answers_objects"]:
                answer_obj = data["answers_objects"][0]
                if "spans" in answer_obj and answer_obj["spans"]:
                    # Take the first non-symbol answer
                    for span in answer_obj["spans"]:
                        if len(span) > 2 and not span.startswith("🚁"):  # Skip emoji answers
                            answer = span
                            break
                    if not answer and answer_obj["spans"]:
                        answer = answer_obj["spans"][0]  # Fallback to first answer
            
            # Use existing question_id if available, otherwise generate
            question_id = data.get("question_id", str(uuid.uuid4()))
            
            return {
                "id": question_id,
                "question": question,
                "answer": answer,
                "source": filename
            }
            
        except Exception as e:
            print(f"[WARNING] Error extracting Trivia format: {str(e)}")
            return None
    
    def _extract_from_json(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract from JSON files (QA-DSML format)"""
        extracted_data = []
        filename = file_path.name
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle if it's a list of objects
            if isinstance(data, list):
                for i, item in enumerate(data):
                    entry = self._process_json_entry(item, filename, i)
                    if entry:
                        extracted_data.append(entry)
            else:
                # Handle single object
                entry = self._process_json_entry(data, filename, 0)
                if entry:
                    extracted_data.append(entry)
                    
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in {filename}: {str(e)}")
            return []
        except Exception as e:
            print(f"[ERROR] Error processing JSON {filename}: {str(e)}")
            return []
        
        return extracted_data
    
    def _process_json_entry(self, data: Dict, filename: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single JSON entry (QA-DSML format)"""
        try:
            # QA-DSML format has Question and Answer fields
            question = data.get("Question", "")
            answer = data.get("Answer", "")
            
            # Generate unique ID
            entry_id = data.get("Q_Id", str(uuid.uuid4()))
            
            return {
                "id": str(entry_id),
                "question": question,
                "answer": answer,
                "source": filename
            }
            
        except Exception as e:
            print(f"[WARNING] Error processing JSON entry {index}: {str(e)}")
            return None
    
    def _extract_from_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract from Parquet files (non-AmbigQA)"""
        extracted_data = []
        filename = file_path.name
        
        try:
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            for index, row in df.iterrows():
                entry = self._process_parquet_row(row, filename, index)
                if entry:
                    extracted_data.append(entry)
                    
        except Exception as e:
            print(f"[ERROR] Error processing Parquet {filename}: {str(e)}")
            return []
        
        return extracted_data
    
    def _process_parquet_row(self, row: pd.Series, filename: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single Parquet row"""
        try:
            # Try to identify question and answer columns
            row_dict = row.to_dict()
            
            # Look for common question/answer column patterns
            question = ""
            answer = ""
            
            # Check for common column names
            for col, value in row_dict.items():
                col_lower = str(col).lower()
                if any(keyword in col_lower for keyword in ['question', 'query', 'input', 'prompt']):
                    question = str(value) if pd.notna(value) else ""
                elif any(keyword in col_lower for keyword in ['answer', 'response', 'output', 'target']):
                    answer = str(value) if pd.notna(value) else ""
            
            # Fallback: if no clear question/answer columns, try first two string columns
            if not question or not answer:
                string_cols = []
                for col, value in row_dict.items():
                    if isinstance(value, str) and len(str(value).strip()) > 10:
                        string_cols.append((col, value))
                
                if len(string_cols) >= 2:
                    question = string_cols[0][1]
                    answer = string_cols[1][1]
                elif len(string_cols) == 1:
                    question = string_cols[0][1]
                    answer = "No answer available"
            
            if question.strip():
                return {
                    "id": str(uuid.uuid4()),
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "source": filename
                }
            
            return None
            
        except Exception as e:
            print(f"[WARNING] Error processing Parquet row {index}: {str(e)}")
            return None
    
    def _save_extracted_data(self, original_file: Path, extracted_data: List[Dict[str, Any]]) -> str:
        """Save extracted data to a new JSON file in the extracted folder"""
        # Create output filename
        base_name = original_file.stem
        output_filename = f"{base_name}_extracted.json"
        output_path = self.extracted_dir / output_filename
        
        # Save as pretty-printed JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        return f"extracted/{output_filename}"
    
    def _print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*80)
        print("UNIFIED DATASET EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total files processed: {len(self.processed_files) + len(self.failed_files)}")
        print(f"Successfully processed: {len(self.processed_files)}")
        print(f"Failed: {len(self.failed_files)}")
        
        if self.processed_files:
            print(f"\n✓ Successfully processed files:")
            total_entries = 0
            for filename, count, output_file in self.processed_files:
                print(f"  - {filename}: {count} entries → {output_file}")
                total_entries += count
            print(f"\nTotal entries extracted: {total_entries}")
        
        if self.failed_files:
            print(f"\n✗ Failed files:")
            for filename, error in self.failed_files:
                print(f"  - {filename}: {error}")
        
        print("="*80)


def main():
    """Main function to run the unified extraction process"""
    parser = argparse.ArgumentParser(description="Extract all dataset files into standardized JSON format")
    parser.add_argument(
        '--include-ambigqa',
        action='store_true',
        default=True,
        help='Include AmbigQA files with special array-to-string conversion (default: True)'
    )
    parser.add_argument(
        '--exclude-ambigqa',
        action='store_true',
        default=False,
        help='Exclude AmbigQA files from processing'
    )
    
    args = parser.parse_args()
    
    # Determine whether to include AmbigQA
    include_ambigqa = args.include_ambigqa and not args.exclude_ambigqa
    
    try:
        extractor = UnifiedDatasetExtractor()
        extractor.extract_all_datasets(include_ambigqa=include_ambigqa)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Extraction process interrupted by user")
    except Exception as e:
        print(f"\n[FATAL ERROR] Extraction process failed: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())