#!/usr/bin/env python3
"""
Prepare mixed training dataset for adversarial fine-tuning.
Combines clean SQuAD data with adversarial AddSent data.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def load_jsonl(path):
    """Load JSONL file"""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

def save_jsonl(examples, path):
    """Save to JSONL file"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')

def create_mixed_dataset(squad_path, addsent_path, output_path, 
                        squad_ratio=0.80, addsent_ratio=0.20, seed=42):
    """
    Create mixed training dataset.
    
    Args:
        squad_path: Path to clean SQuAD data
        addsent_path: Path to adversarial AddSent data
        output_path: Where to save mixed dataset
        squad_ratio: Proportion of SQuAD examples (default 0.80 = 80%)
        addsent_ratio: Proportion of AddSent examples (default 0.20 = 20%)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    print("=" * 60)
    print("Creating Mixed Training Dataset")
    print("=" * 60)
    
    # Load datasets
    print(f"\nLoading SQuAD from: {squad_path}")
    squad_data = load_jsonl(squad_path)
    print(f"  Loaded {len(squad_data)} examples")
    
    print(f"\nLoading AddSent from: {addsent_path}")
    addsent_data = load_jsonl(addsent_path)
    print(f"  Loaded {len(addsent_data)} examples")
    
    # Calculate target sizes
    # Use SQuAD size as base
    total_size = len(squad_data)
    target_squad = int(total_size * squad_ratio)
    target_addsent = int(total_size * addsent_ratio)
    
    print(f"\nTarget distribution (total ~{target_squad + target_addsent}):")
    print(f"  SQuAD (clean): {target_squad} ({squad_ratio*100:.0f}%)")
    print(f"  AddSent (adversarial): {target_addsent} ({addsent_ratio*100:.0f}%)")
    
    # Sample from each dataset
    print("\nSampling examples...")
    
    # Sample SQuAD
    if len(squad_data) >= target_squad:
        squad_sample = random.sample(squad_data, target_squad)
    else:
        squad_sample = squad_data
        print(f"  Warning: Using all {len(squad_data)} SQuAD examples (less than target)")
    
    # Sample AddSent
    if len(addsent_data) >= target_addsent:
        addsent_sample = random.sample(addsent_data, target_addsent)
    else:
        addsent_sample = addsent_data
        print(f"  Warning: Using all {len(addsent_data)} AddSent examples (less than target)")
    
    # Combine and shuffle
    mixed_data = squad_sample + addsent_sample
    random.shuffle(mixed_data)
    
    print(f"\nFinal dataset size: {len(mixed_data)} examples")
    print(f"  SQuAD: {len(squad_sample)} ({len(squad_sample)/len(mixed_data)*100:.1f}%)")
    print(f"  AddSent: {len(addsent_sample)} ({len(addsent_sample)/len(mixed_data)*100:.1f}%)")
    
    # Save
    print(f"\nSaving to: {output_path}")
    save_jsonl(mixed_data, output_path)
    
    print("\n✅ Mixed dataset created successfully!")
    
    # Show sample
    print("\n" + "=" * 60)
    print("Sample Examples")
    print("=" * 60)
    
    # Show one SQuAD example
    squad_ex = [ex for ex in mixed_data if ex['id'] in [s['id'] for s in squad_sample]][0]
    print("\n--- SQuAD Example ---")
    print(f"ID: {squad_ex['id']}")
    print(f"Question: {squad_ex['question']}")
    print(f"Answer: {squad_ex['answers']['text'][0]}")
    print(f"Context (first 150 chars): {squad_ex['context'][:150]}...")
    
    # Show one AddSent example
    addsent_ex = [ex for ex in mixed_data if ex['id'] in [s['id'] for s in addsent_sample]][0]
    print("\n--- AddSent Example ---")
    print(f"ID: {addsent_ex['id']}")
    print(f"Question: {addsent_ex['question']}")
    print(f"Answer: {addsent_ex['answers']['text'][0]}")
    print(f"Context (first 150 chars): {addsent_ex['context'][:150]}...")
    
    return mixed_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare mixed training dataset for adversarial fine-tuning"
    )
    parser.add_argument('--squad_path', type=str, default='./data/squad.jsonl',
                       help='Path to SQuAD training data')
    parser.add_argument('--addsent_path', type=str, default='./data/addsent_adversarial.jsonl',
                       help='Path to AddSent adversarial data')
    parser.add_argument('--output_path', type=str, default='./data/mixed_training.jsonl',
                       help='Output path for mixed dataset')
    parser.add_argument('--squad_ratio', type=float, default=0.80,
                       help='Ratio of SQuAD examples (default: 0.80)')
    parser.add_argument('--addsent_ratio', type=float, default=0.20,
                       help='Ratio of AddSent examples (default: 0.20)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    create_mixed_dataset(
        args.squad_path,
        args.addsent_path,
        args.output_path,
        args.squad_ratio,
        args.addsent_ratio,
        args.seed
    )
