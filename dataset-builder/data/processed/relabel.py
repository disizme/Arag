#!/usr/bin/env python3
import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.adaptive_rag_data.aggregation.label_model import build_label_matrix, aggregate_probabilities

def analyze_json_file(file_path):
    """Count elements and compute statistics for source, score, and reason fields."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Error: Expected list format in {file_path}")
            return
        
        print(f"File: {file_path}")
        print(f"Total elements: {len(data):,}")
        print("=" * 60)
        scores = []

        # Process each element
        dom_labelers = ["lexicon", "embed", "bm25_ratio", "cross_encoder"]
        hall_labelers = ["factual_precision", "obscurity", "complexity"]
        labelers = []
        if 'lexicon' in data[0]['votes']:
            print("Domain relevance")
            labelers = dom_labelers
        elif 'factual_precision' in data[0]['votes']:
            print("Hallucination")
            labelers = hall_labelers
        
        L = build_label_matrix([item["votes"] for item in data], labelers)
        p = aggregate_probabilities(L, seed=int(42), labeler_names=labelers)
        
        for i, item in enumerate(data):
            item["score"] = round(float(p[i]), 2)
            scores.append(item["score"])
        
        print("\nSCORE FIELD STATISTICS:")
        print("-" * 40)
        print(f"Elements with scores: {len(scores):,}")
        print(f"Min score: {min(scores):.3f}")
        print(f"Max score: {max(scores):.3f}")
        print(f"Average score: {sum(scores)/len(scores):.3f}")
        print(f"Median score: {sorted(scores)[len(scores)//2]:.3f}")
        
        # Score distribution
        score_ranges = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, 
            "0.6-0.8": 0, "0.8-1.0": 0
        }
        
        for score in scores:
            if score < 0.2:
                score_ranges["0.0-0.2"] += 1
            elif score < 0.4:
                score_ranges["0.2-0.4"] += 1
            elif score < 0.6:
                score_ranges["0.4-0.6"] += 1
            elif score < 0.8:
                score_ranges["0.6-0.8"] += 1
            else:
                score_ranges["0.8-1.0"] += 1
        
        print("\nScore Distribution:")
        for range_name, count in score_ranges.items():
            percentage = (count / len(scores)) * 100
            print(f"  {range_name}: {count:,} ({percentage:.1f}%)")
    
                # Save to same file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Updated scores saved to {file_path}")
    
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python count_elements.py <json_file>")
        print("Example: python count_elements.py hallucination_dataset.json")
        print()
        print("Counts elements and computes statistics for source, score, and reason fields.")
        return
    
    file_path = sys.argv[1]
    analyze_json_file(file_path)

if __name__ == "__main__":
    main()