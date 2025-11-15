#!/usr/bin/env python3
"""
Convert AddSent dataset from nested SQuAD JSON to flat JSONL format
"""

import json
from pathlib import Path

def convert_addsent_to_jsonl(input_file="./data/addsent.json", 
                              output_file="./data/addsent.jsonl"):
    """
    Convert AddSent SQuAD-format JSON to JSONL
    """
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    print("Converting to JSONL format...")
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in data['data']:
            title = article['title']
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    # Fix answers format: convert list of dicts to dict of lists
                    answers = qa['answers']
                    if isinstance(answers, list):
                        # Convert from [{'text': '...', 'answer_start': ...}, ...]
                        # to {'text': ['...', ...], 'answer_start': [..., ...]}
                        fixed_answers = {
                            'text': [ans['text'] for ans in answers],
                            'answer_start': [ans['answer_start'] for ans in answers]
                        }
                    else:
                        # Already in correct format
                        fixed_answers = answers
                    
                    # Create flattened format
                    item = {
                        'id': qa['id'],
                        'title': title,
                        'context': context,
                        'question': qa['question'],
                        'answers': fixed_answers
                    }
                    f.write(json.dumps(item) + '\n')
                    count += 1
    
    print(f"\n✅ Converted {count} examples to {output_path}")
    print(f"\nNow you can evaluate with:")
    print(f"  python3 run.py --do_eval --task qa \\")
    print(f"    --dataset {output_path} \\")
    print(f"    --model ./trained_model/ \\")
    print(f"    --output_dir ./eval_addsent/")

if __name__ == "__main__":
    convert_addsent_to_jsonl()
