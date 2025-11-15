#!/usr/bin/env python3
"""
Convert SQuAD format JSON to a format compatible with HuggingFace datasets JSON loader
Flattens the nested structure into individual examples
"""

import json
import sys
from pathlib import Path

def flatten_squad_to_jsonl(input_file, output_file):
    """
    Convert SQuAD format to JSONL with one example per line
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    # Flatten the nested structure
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                example = {
                    'id': qa['id'],
                    'title': article.get('title', ''),
                    'context': context,
                    'question': qa['question'],
                    'answers': qa['answers']
                }
                examples.append(example)
    
    # Write as JSONL (one JSON object per line)
    print(f"Writing {len(examples)} examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Converted {len(examples)} examples")
    return len(examples)

def main():
    input_file = Path("./data/squad_adversarial_strong.json")
    output_file = Path("./data/squad_adversarial_strong.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    try:
        num_examples = flatten_squad_to_jsonl(input_file, output_file)
        print(f"\n✅ Success!")
        print(f"   Input:  {input_file}")
        print(f"   Output: {output_file}")
        print(f"   Examples: {num_examples}")
        print(f"\nUse with: --dataset {output_file}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
