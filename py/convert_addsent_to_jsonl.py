#!/usr/bin/env python3
"""
Convert the original AddSent dev-v1.1.json to JSONL format
"""

import json
import sys
from pathlib import Path

def convert_file(input_file, output_file):
    """
    Convert SQuAD format JSON to JSONL
    """
    print(f"Loading {input_file}...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        return 0
    
    examples = []
    
    # Flatten the nested structure
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                # Ensure answers are in the correct format
                answers = qa['answers']
                # Convert to the format expected by SQuAD
                if isinstance(answers, dict):
                    # Already in correct format with 'text' and 'answer_start' lists
                    formatted_answers = answers
                else:
                    # If it's a list of answer dicts, convert it
                    formatted_answers = {
                        'text': [ans['text'] for ans in answers],
                        'answer_start': [ans['answer_start'] for ans in answers]
                    }
                
                example = {
                    'id': qa['id'],
                    'title': article.get('title', ''),
                    'context': context,
                    'question': qa['question'],
                    'answers': formatted_answers
                }
                examples.append(example)
    
    # Write as JSONL
    print(f"Writing {len(examples)} examples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    return len(examples)

def main():
    # Convert the original AddSent file
    input_file = Path("./data/dev-v1.1.json")
    output_file = Path("./data/addsent_adversarial.jsonl")
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("\nThis should be the original AddSent adversarial dataset.")
        print("Make sure you have dev-v1.1.json in the data/ directory.")
        return 1
    
    try:
        num_examples = convert_file(input_file, output_file)
        
        if num_examples > 0:
            print(f"\n✅ Success!")
            print(f"   Input:  {input_file}")
            print(f"   Output: {output_file}")
            print(f"   Examples: {num_examples}")
            print(f"\nThis is the ORIGINAL AddSent adversarial dataset!")
            print(f"Use with: --dataset {output_file}")
            return 0
        else:
            print(f"❌ No examples found in {input_file}")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
