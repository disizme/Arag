#!/usr/bin/env python3
"""
Dataset Collection Script

Collects samples from all extracted Q&A JSON files, shuffles them, and creates
separate datasets for hallucination and specialization training.

Usage:
    python collect_datasets.py [--samples_per_file N] [--output_dir PATH]
"""

import json
import random
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_file_info(file_path: Path) -> Tuple[List[Dict], int]:
    """
    Load a JSON file and return its data and size.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Tuple of (data, size)
    """
    try:
        logger.info(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"Unexpected format in {file_path.name}, skipping")
            return [], 0
        
        logger.info(f"Loaded {file_path.name}: {len(data)} items")
        return data, len(data)
        
    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return [], 0

def sample_from_file(data: List[Dict], file_name: str, sample_size: int) -> List[Dict]:
    """
    Sample items from file data.
    
    Args:
        data: The file data
        file_name: Name of the file (for logging)
        sample_size: Number of samples to take
        
    Returns:
        List of sampled items
    """
    if len(data) <= sample_size:
        # Take all if file has fewer samples than requested
        sampled = data.copy()
    else:
        # Randomly sample
        sampled = random.sample(data, sample_size)
    
    logger.info(f"Sampled {len(sampled)} items from {file_name} (total: {len(data)})")
    return sampled

def collect_all_datasets(
    extracted_dir: Path,
    samples_per_dataset: int = 1500,  # Exact samples per output dataset
    max_per_file: int = 300,  # Maximum samples from each dataset file in final output
    output_dir: Path = None,
    existing_datasets: Tuple[List[Dict], List[Dict]] = None  # Existing datasets to avoid duplicates
) -> Tuple[List[Dict], List[Dict]]:
    """
    Collect samples from all extracted files and create two datasets with exactly the specified number of samples each.
    
    Args:
        extracted_dir: Directory containing extracted JSON files
        samples_per_dataset: Exact number of samples per output dataset (default: 1500)
        max_per_file: Maximum samples from each dataset file in final output (default: 300)
        output_dir: Directory to save the collected datasets
        existing_datasets: Tuple of (existing_hallucination, existing_specialization) to avoid duplicates
        
    Returns:
        Tuple of (hallucination_dataset, specialization_dataset) each with exactly samples_per_dataset samples
    """
    
    if output_dir is None:
        output_dir = extracted_dir.parent / "collected_datasets"
    
    output_dir.mkdir(exist_ok=True)
    
    # Create set of existing sample IDs to avoid duplicates
    existing_ids = set()
    if existing_datasets:
        existing_hallucination, existing_specialization = existing_datasets
        logger.info(f"Found existing datasets: {len(existing_hallucination)} hallucination, {len(existing_specialization)} specialization samples")
        
        # Collect IDs from both existing datasets
        for sample in existing_hallucination:
            if 'id' in sample:
                existing_ids.add(sample['id'])
            else:
                # Create hash-based ID for samples without ID
                content = f"{sample.get('question', '')}{sample.get('answer', '')}"
                existing_ids.add(hashlib.md5(content.encode()).hexdigest())
        
        for sample in existing_specialization:
            if 'id' in sample:
                existing_ids.add(sample['id'])
            else:
                # Create hash-based ID for samples without ID
                content = f"{sample.get('question', '')}{sample.get('answer', '')}"
                existing_ids.add(hashlib.md5(content.encode()).hexdigest())
        
        logger.info(f"Will avoid {len(existing_ids)} existing sample IDs")
    
    # Find all extracted JSON files
    json_files = list(extracted_dir.glob("*_extracted.json"))
    
    if not json_files:
        logger.error("No extracted JSON files found!")
        return [], []
    
    logger.info(f"Found {len(json_files)} extracted files")
    
    # First pass: Load all files and get their sizes
    logger.info("Analyzing file sizes for proportional sampling...")
    file_data = {}
    total_available = 0
    
    for file_path in json_files:
        data, size = load_file_info(file_path)
        if size > 0:
            file_data[file_path] = data
            total_available += size
            logger.info(f"  - {file_path.name}: {size:,} items")
    
    if not file_data:
        logger.error("No valid data found in any files!")
        return [], []
    
    logger.info(f"Total available samples: {total_available:,}")
    
    # Calculate how many samples we need total (for both datasets with overlap)
    # We need enough samples to create two datasets of samples_per_dataset each with minimal overlap
    overlap_percentage = 0.10  # 10% overlap between datasets
    overlap_size = int(samples_per_dataset * overlap_percentage)
    unique_per_dataset = samples_per_dataset - overlap_size
    total_unique_needed = unique_per_dataset * 2  # Unique samples for both datasets
    total_samples_needed = total_unique_needed + overlap_size  # Total unique samples + shared samples
    
    logger.info(f"Target samples per dataset: {samples_per_dataset:,}")
    logger.info(f"Overlap between datasets: {overlap_size:,} samples ({overlap_percentage:.0%})")
    logger.info(f"Total samples needed: {total_samples_needed:,}")
    
    # Calculate boosted sampling for smaller files
    logger.info("Calculating sample sizes with boost for smaller files...")
    sample_plan = {}
    
    # Sort files by size to apply different strategies
    files_by_size = sorted(file_data.items(), key=lambda x: len(x[1]))
    
    for file_path, data in files_by_size:
        file_size = len(data)
        
        # Calculate base proportional samples
        base_proportion = file_size / total_available
        base_samples = int(total_samples_needed * base_proportion)
        
        # Apply boosting strategy based on file size (ordered from smallest to largest)
        if file_size < 1000:
            # Very small files: take 100%
            actual_samples = 1
            boost_note = f" (boosted 100% for very small file)"
            
        elif file_size < 5000:
            # Small files: take 60-75% to ensure good representation
            actual_samples = min(base_samples, int(file_size * 0.6))
            boost_note = f" (boosted 60% for small file)"
            
        elif file_size < 10000:
            # Medium-large files: slight reduction (max 30% of file)
            actual_samples = min(base_samples, int(file_size * 0.3))
            boost_note = " (reduced to 30% for medium-large file)"
            
        elif file_size < 20000:
            # Large files: moderate reduction (max 10% of file)
            actual_samples = min(base_samples, max_per_file, int(file_size * 0.05))
            boost_note = " (reduced to 10% for large file)"
            
        # Apply absolute maximum limit per file (default: max_per_file samples)
        file_max_limit = max_per_file
            
        actual_samples = min(actual_samples, file_size, file_max_limit)
        sample_plan[file_path] = actual_samples
        
        logger.info(f"  - {file_path.name}: {actual_samples:,} samples "
                   f"(proportion: {base_proportion:.1%}, available: {file_size:,}){boost_note}")
    
    # Verify we have enough total samples planned
    total_planned = sum(sample_plan.values())
    logger.info(f"Total planned samples: {total_planned:,} (need: {total_samples_needed:,})")
    
    # If we don't have enough, boost smaller files further
    if total_planned < total_samples_needed:
        shortage = total_samples_needed - total_planned
        logger.info(f"Need to add {shortage:,} more samples. Boosting smaller files...")
        
        # Sort by file size and boost smallest files first
        for file_path, data in files_by_size:
            if shortage <= 0:
                break
                
            file_size = len(data)
            current_samples = sample_plan[file_path]
            
            # Calculate how many more we can take from this file (respecting max limit)
            file_max_limit = max_per_file
                
            max_from_file = min(file_max_limit, file_size)  # Respect limit
            max_additional = min(shortage, max_from_file - current_samples, file_size // 2)
            
            if max_additional > 0:
                new_total = current_samples + max_additional
                sample_plan[file_path] = min(new_total, file_max_limit)  # Enforce limit
                actual_added = sample_plan[file_path] - current_samples
                shortage -= actual_added
                logger.info(f"  + Boosted {file_path.name}: +{actual_added:,} samples (total: {sample_plan[file_path]:,})")
    
    # Collect samples from all files, filtering out existing ones
    all_samples = []
    for file_path, planned_samples in sample_plan.items():
        data = file_data[file_path]
        
        # Filter out existing samples
        available_data = []
        for sample in data:
            sample_id = sample.get('id')
            if not sample_id:
                # Create hash-based ID for samples without ID
                content = f"{sample.get('question', '')}{sample.get('answer', '')}"
                sample_id = hashlib.md5(content.encode()).hexdigest()
            
            if sample_id not in existing_ids:
                available_data.append(sample)
        
        if len(available_data) == 0:
            logger.warning(f"All samples from {file_path.name} already exist in datasets, skipping")
            continue
        
        logger.info(f"File {file_path.name}: {len(available_data)} available samples (after filtering {len(data) - len(available_data)} existing ones)")
        
        samples = sample_from_file(available_data, file_path.name, planned_samples)
        all_samples.extend(samples)
    
    logger.info(f"Total samples collected: {len(all_samples):,}")
    
    # If we don't have enough samples, collect more from largest files
    if len(all_samples) < total_samples_needed:
        shortage = total_samples_needed - len(all_samples)
        logger.info(f"Need {shortage:,} more samples. Collecting from largest files...")
        
        # Sort files by remaining capacity
        remaining_capacity = []
        for file_path, data in file_data.items():
            used = sample_plan[file_path]
            available = len(data) - used
            if available > 0:
                remaining_capacity.append((file_path, available, data))
        
        remaining_capacity.sort(key=lambda x: x[1], reverse=True)  # Sort by available capacity
        
        for file_path, available, data in remaining_capacity:
            if shortage <= 0:
                break
            
            additional_samples = min(shortage, available)
            used_samples = sample_plan[file_path]
            
            # Get additional samples from this file (avoiding already selected ones)
            remaining_data = data[used_samples:used_samples + additional_samples]
            all_samples.extend(remaining_data)
            
            shortage -= additional_samples
            logger.info(f"  + Added {additional_samples:,} more samples from {file_path.name}")
    
    # If we have too many samples, randomly trim to exact amount needed
    if len(all_samples) > total_samples_needed:
        logger.info(f"Trimming from {len(all_samples):,} to {total_samples_needed:,} samples")
        all_samples = random.sample(all_samples, total_samples_needed)
    
    if not all_samples:
        logger.error("No samples collected!")
        return [], []
    
    if len(all_samples) < total_samples_needed:
        logger.warning(f"Only collected {len(all_samples):,} samples, need {total_samples_needed:,}. Adjusting overlap strategy...")
        # Reduce overlap if we don't have enough samples
        available_samples = len(all_samples)
        max_possible_per_dataset = available_samples // 2
        if max_possible_per_dataset < samples_per_dataset:
            logger.warning(f"Cannot create datasets of {samples_per_dataset:,} samples each. Maximum possible: {max_possible_per_dataset:,}")
            samples_per_dataset = max_possible_per_dataset
            overlap_size = int(samples_per_dataset * 0.05)  # Reduce overlap to 5%
    
    logger.info(f"Final samples collected: {len(all_samples):,}")
    
    # Shuffle all samples first
    logger.info("Shuffling all samples...")
    random.shuffle(all_samples)
    
    # Create two datasets with exactly samples_per_dataset samples each
    logger.info(f"Creating datasets with exactly {samples_per_dataset:,} samples each...")
    logger.info(f"Overlap: {overlap_size:,} samples ({overlap_size/samples_per_dataset:.1%})")
    
    unique_per_dataset = samples_per_dataset - overlap_size
    
    # Create shared pool for overlap
    shared_pool = all_samples[:overlap_size]
    remaining_samples = all_samples[overlap_size:]
    
    # Create unique pools for each dataset
    hallucination_unique = remaining_samples[:unique_per_dataset]
    specialization_unique = remaining_samples[unique_per_dataset:unique_per_dataset*2]
    
    # Combine shared + unique for each dataset
    hallucination_dataset = shared_pool + hallucination_unique
    specialization_dataset = shared_pool + specialization_unique
    
    # Ensure exact sample count (trim or pad if necessary)
    if len(hallucination_dataset) > samples_per_dataset:
        hallucination_dataset = hallucination_dataset[:samples_per_dataset]
    elif len(hallucination_dataset) < samples_per_dataset:
        # Pad with additional samples if needed
        additional_needed = samples_per_dataset - len(hallucination_dataset)
        additional_samples = remaining_samples[unique_per_dataset*2:unique_per_dataset*2 + additional_needed]
        hallucination_dataset.extend(additional_samples)
    
    if len(specialization_dataset) > samples_per_dataset:
        specialization_dataset = specialization_dataset[:samples_per_dataset]
    elif len(specialization_dataset) < samples_per_dataset:
        # Pad with additional samples if needed
        additional_needed = samples_per_dataset - len(specialization_dataset)
        start_idx = unique_per_dataset*2 + (samples_per_dataset - len(hallucination_dataset))
        additional_samples = remaining_samples[start_idx:start_idx + additional_needed]
        specialization_dataset.extend(additional_samples)
    
    # Final shuffle of each dataset
    random.shuffle(hallucination_dataset)
    random.shuffle(specialization_dataset)
    
    logger.info(f"✅ Hallucination dataset: {len(hallucination_dataset):,} samples")
    logger.info(f"✅ Specialization dataset: {len(specialization_dataset):,} samples")
    
    # Determine file naming - if existing datasets provided, create new numbered files
    if existing_datasets:
        # Find next available file number
        existing_hall_files = list(output_dir.glob("hallucination_dataset_*.json"))
        existing_spec_files = list(output_dir.glob("specialization_dataset_*.json"))
        
        next_num = 1
        if existing_hall_files or existing_spec_files:
            # Extract numbers from existing files
            numbers = []
            for f in existing_hall_files + existing_spec_files:
                try:
                    num = int(f.stem.split('_')[-1])
                    numbers.append(num)
                except:
                    pass
            next_num = max(numbers, default=0) + 1
        
        hall_path = output_dir / f"hallucination_dataset_{next_num:02d}.json"
        spec_path = output_dir / f"specialization_dataset_{next_num:02d}.json"
    else:
        hall_path = output_dir / "hallucination_dataset.json"
        spec_path = output_dir / "specialization_dataset.json"
    
    # Save hallucination dataset
    logger.info(f"Saving hallucination dataset to {hall_path}")
    with open(hall_path, 'w', encoding='utf-8') as f:
        json.dump(hallucination_dataset, f, indent=2, ensure_ascii=False)
    
    # Save specialization dataset
    logger.info(f"Saving specialization dataset to {spec_path}")
    with open(spec_path, 'w', encoding='utf-8') as f:
        json.dump(specialization_dataset, f, indent=2, ensure_ascii=False)
    
    # Generate statistics for both datasets
    stats = generate_comprehensive_statistics(
        all_samples, hallucination_dataset, specialization_dataset, json_files
    )
    
    # Save statistics
    stats_path = output_dir / "collection_statistics.json"
    logger.info(f"Saving statistics to {stats_path}")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print_comprehensive_summary(stats, output_dir)
    
    return hallucination_dataset, specialization_dataset

def generate_comprehensive_statistics(
    all_samples: List[Dict], 
    hallucination_dataset: List[Dict], 
    specialization_dataset: List[Dict], 
    source_files: List[Path]
) -> Dict:
    """Generate comprehensive statistics about all collected datasets."""
    
    def analyze_dataset(samples, name):
        """Helper function to analyze a single dataset"""
        source_datasets = {}
        question_lengths = []
        answer_lengths = []
        
        for sample in samples:
            source_dataset = sample.get('source', 'unknown')
            source_datasets[source_dataset] = source_datasets.get(source_dataset, 0) + 1
            
            if 'question' in sample:
                question_lengths.append(len(str(sample['question'])))
            if 'answer' in sample:
                answer_lengths.append(len(str(sample['answer'])))
        
        return {
            'total_samples': len(samples),
            'samples_by_source_dataset': source_datasets,
            'question_length_stats': {
                'avg': sum(question_lengths) / len(question_lengths) if question_lengths else 0,
                'min': min(question_lengths) if question_lengths else 0,
                'max': max(question_lengths) if question_lengths else 0
            },
            'answer_length_stats': {
                'avg': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0,
                'min': min(answer_lengths) if answer_lengths else 0,
                'max': max(answer_lengths) if answer_lengths else 0
            }
        }
    
    # Calculate overlap between datasets
    hall_ids = set(sample.get('id', str(i)) for i, sample in enumerate(hallucination_dataset))
    spec_ids = set(sample.get('id', str(i)) for i, sample in enumerate(specialization_dataset))
    overlap_count = len(hall_ids.intersection(spec_ids))
    overlap_percentage = (overlap_count / len(hall_ids)) * 100 if hall_ids else 0
    
    stats = {
        'total_collected_samples': len(all_samples),
        'source_files_processed': len(source_files),
        'overlap_stats': {
            'overlapping_samples': overlap_count,
            'overlap_percentage': round(overlap_percentage, 2)
        },
        'hallucination_dataset': analyze_dataset(hallucination_dataset, 'hallucination'),
        'specialization_dataset': analyze_dataset(specialization_dataset, 'specialization'),
        'all_samples': analyze_dataset(all_samples, 'all_samples')
    }
    
    return stats

def print_comprehensive_summary(stats: Dict, output_dir: Path):
    """Print a summary of the collection process."""
    
    print("\n" + "="*70)
    print("📊 DATASET COLLECTION SUMMARY")
    print("="*70)
    
    print(f"✅ Total samples collected: {stats['total_collected_samples']:,}")
    print(f"📁 Source files processed: {stats['source_files_processed']}")
    print(f"💾 Output directory: {output_dir}")
    
    # Overlap statistics
    overlap = stats['overlap_stats']
    print(f"\n🔗 Dataset Overlap:")
    print(f"   Overlapping samples: {overlap['overlapping_samples']:,}")
    print(f"   Overlap percentage: {overlap['overlap_percentage']:.1f}%")
    
    # Individual dataset stats
    for dataset_name in ['hallucination_dataset', 'specialization_dataset']:
        dataset_stats = stats[dataset_name]
        display_name = dataset_name.replace('_', ' ').title()
        
        print(f"\n📋 {display_name}:")
        print(f"   Total samples: {dataset_stats['total_samples']:,}")
        
        print(f"   📊 By original dataset:")
        for dataset, count in sorted(dataset_stats['samples_by_source_dataset'].items()):
            print(f"      {dataset}: {count:,} samples")
        
        print(f"   📏 Question length - Avg: {dataset_stats['question_length_stats']['avg']:.1f} chars")
        print(f"   📏 Answer length - Avg: {dataset_stats['answer_length_stats']['avg']:.1f} chars")
    
    print(f"\n💾 Generated files:")
    print(f"   📄 hallucination_dataset.json")
    print(f"   📄 specialization_dataset.json") 
    print(f"   📄 collection_statistics.json")
    

    print("="*70)

def load_existing_datasets(output_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """
    Load existing datasets to avoid duplicates.
    
    Args:
        output_dir: Directory containing existing datasets
        
    Returns:
        Tuple of (existing_hallucination, existing_specialization) datasets
    """
    existing_hallucination = []
    existing_specialization = []
    
    # Find all existing dataset files
    hall_files = list(output_dir.glob("processed_hallucination_dataset*.json"))
    spec_files = list(output_dir.glob("processed_specialization_dataset*.json"))
    
    # Load all hallucination datasets
    for file_path in hall_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_hallucination.extend(data)
                logger.info(f"Loaded {len(data)} samples from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Load all specialization datasets
    for file_path in spec_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                existing_specialization.extend(data)
                logger.info(f"Loaded {len(data)} samples from {file_path.name}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return existing_hallucination, existing_specialization

def main():
    """Main function to run the collection script."""
    
    parser = argparse.ArgumentParser(description="Collect Q&A samples from extracted files")
    parser.add_argument(
        '--samples_per_dataset', 
        type=int, 
        default=500,
        help='Number of NEW samples per output dataset (default: 500)'
    )
    parser.add_argument(
        '--avoid_duplicates',
        action='store_true',
        default=True,
        help='Load existing datasets and avoid duplicates (default: False)'
    )
    parser.add_argument(
        '--max_per_file', 
        type=int, 
        default=1000,
        help='Maximum samples from each dataset file in final output (default: 300)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Output directory for collected datasets (default: ../preprocessed_datasets)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=512,
        help='Random seed for reproducible sampling (default: 1024)'
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(args.seed)
    
    # Get directories
    script_dir = Path(__file__).parent
    output_dir = Path(args.output_dir) if args.output_dir else script_dir.parent / "preprocessed_datasets"
    
    logger.info(f"Starting dataset collection...")
    logger.info(f"Samples per dataset: {args.samples_per_dataset}")
    logger.info(f"Max samples per file: {args.max_per_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Avoid duplicates: {args.avoid_duplicates}")
    
    # Load existing datasets if avoiding duplicates
    existing_datasets = None
    if args.avoid_duplicates:
        logger.info("Loading existing datasets to avoid duplicates...")
        existing_datasets = load_existing_datasets(output_dir)
        if existing_datasets[0] or existing_datasets[1]:
            logger.info(f"Found {len(existing_datasets[0])} existing hallucination samples")
            logger.info(f"Found {len(existing_datasets[1])} existing specialization samples")
        else:
            logger.info("No existing datasets found")
    
    # Run collection
    hall_dataset, spec_dataset = collect_all_datasets(
        extracted_dir=script_dir,
        samples_per_dataset=args.samples_per_dataset,
        max_per_file=args.max_per_file,
        output_dir=output_dir,
        existing_datasets=existing_datasets
    )
    
    if hall_dataset and spec_dataset:
        logger.info("✅ Dataset collection completed successfully!")
    else:
        logger.error("❌ Dataset collection failed!")

if __name__ == "__main__":
    main() 