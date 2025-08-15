#!/usr/bin/env python3
"""
Script to sort lexicon terms alphabetically and remove duplicates.
Removes empty lines and sorts terms for better organization.
"""

import argparse
from pathlib import Path


def dedupe_lexicon(input_file: str, output_file: str = None) -> None:
    """
    Sort lexicon terms alphabetically and remove duplicates.
    
    Args:
        input_file: Path to input lexicon file
        output_file: Path to output file (defaults to input_file if None)
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: File {input_file} does not exist")
        return
    
    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Clean and collect all non-empty lines
    clean_lines = []
    original_count = 0
    
    for line in lines:
        # Clean whitespace and skip empty lines
        clean_line = line.strip()
        if not clean_line:
            continue
            
        original_count += 1
        clean_lines.append(clean_line)
    
    # Remove duplicates (case-insensitive) while preserving original case
    seen = set()
    unique_lines = []
    for line in clean_lines:
        if line.lower() not in seen:
            seen.add(line.lower())
            unique_lines.append(line)
    
    # Sort alphabetically (case-insensitive)
    unique_lines.sort(key=str.lower)
    
    # Write sorted and deduplicated file
    output_path = Path(output_file) if output_file else input_path
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in unique_lines:
            f.write(line + '\n')
    
    duplicates_removed = original_count - len(unique_lines)
    print(f"Processed {input_file}")
    print(f"Original terms: {original_count}")
    print(f"Unique terms: {len(unique_lines)}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Terms sorted alphabetically")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Sort lexicon terms alphabetically and remove duplicates")
    parser.add_argument("input_file", help="Input lexicon file path")
    parser.add_argument("-o", "--output", help="Output file path (defaults to input file)")
    parser.add_argument("--backup", action="store_true", help="Create backup of original file")
    
    args = parser.parse_args()
    
    # Create backup if requested
    if args.backup:
        input_path = Path(args.input_file)
        backup_path = input_path.with_suffix(input_path.suffix + '.bak')
        backup_path.write_text(input_path.read_text(encoding='utf-8'), encoding='utf-8')
        print(f"Backup created: {backup_path}")
    
    dedupe_lexicon(args.input_file, args.output)


if __name__ == "__main__":
    main()