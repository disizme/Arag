#!/usr/bin/env python3
"""
Script to process all JSON files in current directory with Ollama model.
Reads all JSON files, processes each question with a predefined prompt,
and stores the model responses back in each JSON file.
"""

import json
import os
import sys
import ollama
from typing import Dict, Any, List


class OllamaProcessor:
    def __init__(self, model_name: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=host)
        
    def chat_with_model(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to Ollama and get response using ollama library."""
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages
            )
            return response['message']['content']
            
        except ollama.ResponseError as e:
            print(f"Ollama response error: {e}")
            return ""
        except Exception as e:
            print(f"Unexpected error: {e}")
            return ""


def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load JSON file and return parsed data."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{file_path}': {e}")
        sys.exit(1)


def save_json_file(file_path: str, data: Dict[str, Any]) -> None:
    """Save data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving file '{file_path}': {e}")
        sys.exit(1)


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """Parse model response to extract score and domain."""
    try:
        # Try to parse as JSON directly
        parsed = json.loads(response_text.strip())
        if isinstance(parsed, dict) and "score" in parsed and "domain" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    
    # Return empty result if parsing fails
    return {"score": None, "domain": None}


def process_questions(data: Dict[str, Any], processor: OllamaProcessor, prompt: str) -> Dict[str, Any]:
    """Process each question in the JSON data with the predefined prompt."""
    
    processed_count = 0
    
    # Initialize conversation with the system prompt once
    messages = [{"role": "system", "content": prompt}]
    
    for i, item in enumerate(data):
        # Look for question field
        question_text = item["question"]

        print(f"Processing question {i+1}/{len(data)}: {question_text[:50]}...")
        
        # Create a fresh conversation for each question with the system prompt
        current_messages = messages + [{"role": "user", "content": question_text}]
        
        # Get response from Ollama
        response = processor.chat_with_model(current_messages)
        
        if response:            # Parse the model response to extract score and domain
            parsed_response = parse_model_response(response)
            
            # Store parsed response in the item
            item["score"] = parsed_response["score"]
            item["domain"] = parsed_response["domain"]
            
            processed_count += 1
            print(f"✓ Processed successfully - Score: {parsed_response['score']}, domain: {parsed_response['domain']}")
    
    print(f"\nProcessed {processed_count}/{len(data)} questions successfully.")
    return data


def get_json_files() -> List[str]:
    """Get all JSON files in current directory """
    json_files = []
    current_dir = os.getcwd()
    
    for filename in os.listdir(current_dir):
        if filename.endswith('.json') :
            json_files.append(filename)
    
    return sorted(json_files)


def main():
    # Default settings
    model_name = "granite3.3:8b"
    host_url = "http://localhost:11434"
    system_prompt = """You are an expert evaluator rating the level of AI/ML domain expertise required to answer questions accurately.
Scoring Guidelines
0.0-0.2: Non-AI/ML topics

General trivia, unrelated domains, random questions

0.3-0.4: Basic AI/ML knowledge

Fundamental definitions, introductory concepts
Simple library usage, basic terminology

0.5-0.6: Intermediate AI/ML understanding

Practical applications, algorithm explanations
Code implementation, optimization techniques

0.7-0.8: Advanced AI/ML expertise

Complex implementations, architectural details
Performance tuning, 

0.9-1.0: Expert-level AI/ML knowledge

Cutting-edge research, experimental techniques, advanced methodologies
Novel architectures, theoretical foundations, advanced research

Domain Categories (exactly 2 words)
AI/ML related: "Basic AI", "Machine Learning", "Deep Learning", "Technical Implementation", "Advanced Research"
Non-AI/ML: "General Knowledge", "Programming Concepts", "Data Science", "Software Engineering", "Random Trivia"
Output Format
{"score": 0.0, "domain": "category name"}

Examples:
question: "what is artificial intelligence" → score: 0.3, domain: "Basic AI"
question: "explain backpropagation algorithm" → score: 0.5, domain : "Machine Learning"
question: "implement transformer attention mechanism" → score: 0.8, domain: "Technical Implementation"

If the question is not related to AI/ML, assign a lower risk (unless it’s ambiguous).
Keep the answer concise and in this specified format {"score":0.0, "domain": "ML Architecture"}.Do not include anything else at all.
if you cannot determine the score, assign score based on AI/ML related keywords in the question.
"""
    
    # Get all JSON files to process
    json_files = get_json_files()
    
    if not json_files:
        print("No JSON files found in current directory")
        return
    
    print(f"Found {len(json_files)} JSON files to process:")
    for filename in json_files:
        print(f"  - {filename}")
    print()
    
    # Initialize Ollama processor
    processor = OllamaProcessor(model_name=model_name, host=host_url)
    
    # Process each JSON file
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] Processing file: {json_file}")
        print("-" * 50)
        
        # Load JSON data
        data = load_json_file(json_file)
        
        # Process questions
        processed_data = process_questions(data, processor, system_prompt)
        
        # Save results (overwrite input file)
        save_json_file(json_file, processed_data)
        
        print(f"✓ Completed processing {json_file}\n")
    
    print(f"Done! Processed {len(json_files)} files successfully.")


if __name__ == "__main__":
    main()