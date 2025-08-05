#!/usr/bin/env python3
import json
import sys
from collections import Counter

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
        
        # Initialize counters
        source_counts = Counter()
        reason_counts = Counter()
        scores = []
        
        # Process each element
        for item in data:
            if isinstance(item, dict):
                # Count sources
                if 'source' in item:
                    source_counts[item['source']] += 1
                
                # Count reasons
                if 'domain' in item:
                    reason_counts[item['domain']] += 1
                
                if 'reason' in item:
                    reason_counts[item['reason']] += 1

                # Collect scores
                if 'score' in item:
                    try:
                        score = float(item['score'])
                        scores.append(score)
                    except (ValueError, TypeError):
                        pass
        
        # Source statistics
        if source_counts:
            print("\nSOURCE FIELD STATISTICS:")
            print("-" * 40)
            print(f"Unique sources: {len(source_counts)}")
            for source, count in source_counts.most_common():
                percentage = (count / len(data)) * 100
                print(f"  {source}: {count:,} ({percentage:.1f}%)")
        
        # Reason statistics
        if reason_counts:
            print("\nREASON FIELD STATISTICS:")
            print("-" * 40)
            print(f"Unique reasons: {len(reason_counts)}")
            print("Top 5 reasons:")
            for reason, count in reason_counts.most_common(5):
                percentage = (count / len(data)) * 100
                print(f"  {reason}: {count:,} ({percentage:.1f}%)")
        
        # Score statistics
        if scores:
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
        
        # Missing field statistics
        missing_reason = sum(1 for item in data if isinstance(item, dict) and 'reason' not in item)
        missing_domain = sum(1 for item in data if isinstance(item, dict) and 'domain' not in item)
        missing_score = sum(1 for item in data if isinstance(item, dict) and 'score' not in item)
        
        if  missing_domain or missing_reason or missing_score:
            print("\nMISSING FIELD STATISTICS:")
            print("-" * 40)
            print(f"Missing domain: {missing_domain:,}")
            print(f"Missing reason: {missing_reason:,}")
            print(f"Missing score: {missing_score:,}")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{file_path}': {e}")
    except Exception as e:
        print(f"Error processing '{file_path}': {e}")

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