#!/usr/bin/env python3
"""
Generate STRONG adversarial examples for SQuAD
Uses question-aware distractors with wrong but plausible answers
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import re

random.seed(42)

# Entity lists for generating wrong answers
PEOPLE = ["John Smith", "Mary Johnson", "David Williams", "Sarah Brown", "Michael Davis",
          "Jennifer Wilson", "James Anderson", "Lisa Martinez", "Robert Taylor", "Emily Moore"]

PLACES = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
          "San Antonio", "San Diego", "Dallas", "San Jose", "London", "Paris", "Tokyo"]

DATES = ["January 15, 2015", "March 22, 2016", "June 10, 2014", "September 5, 2017",
         "December 1, 2013", "April 18, 2015", "July 4, 2016", "October 31, 2014"]

NUMBERS = ["5", "10", "15", "25", "50", "100", "500", "1000", "three", "five", "ten", "twenty"]

ORGANIZATIONS = ["NASA", "FBI", "CIA", "Microsoft", "Google", "Amazon", "IBM", "Intel"]

def extract_answer_type(question, answer):
    """Determine what type of answer the question is asking for"""
    q_lower = question.lower()
    
    # Check question words
    if any(word in q_lower for word in ["who", "whom"]):
        return "PERSON"
    elif any(word in q_lower for word in ["where", "which city", "which country", "which state"]):
        return "PLACE"
    elif any(word in q_lower for word in ["when", "what year", "what date", "which year"]):
        return "DATE"
    elif any(word in q_lower for word in ["how many", "how much", "what number"]):
        return "NUMBER"
    elif any(word in q_lower for word in ["which company", "which organization", "what organization"]):
        return "ORG"
    else:
        return "OTHER"

def generate_wrong_answer(answer_type):
    """Generate a plausible wrong answer based on type"""
    if answer_type == "PERSON":
        return random.choice(PEOPLE)
    elif answer_type == "PLACE":
        return random.choice(PLACES)
    elif answer_type == "DATE":
        return random.choice(DATES)
    elif answer_type == "NUMBER":
        return random.choice(NUMBERS)
    elif answer_type == "ORG":
        return random.choice(ORGANIZATIONS)
    else:
        return random.choice(PEOPLE + PLACES)

def create_adversarial_sentence(question, correct_answer, answer_type):
    """Create an adversarial sentence with a plausible wrong answer"""
    wrong_answer = generate_wrong_answer(answer_type)
    
    # Make sure wrong answer is different from correct answer
    attempts = 0
    while wrong_answer.lower() in correct_answer.lower() and attempts < 10:
        wrong_answer = generate_wrong_answer(answer_type)
        attempts += 1
    
    # Templates that create competing information
    templates = [
        f"However, {wrong_answer} claimed responsibility for the event.",
        f"According to recent reports, {wrong_answer} was actually involved.",
        f"Some sources suggest {wrong_answer} played a key role.",
        f"Initial reports indicated {wrong_answer} was responsible.",
        f"Historians later discovered {wrong_answer} was the actual participant.",
        f"The evidence points to {wrong_answer} as the primary figure.",
        f"Research reveals {wrong_answer} was central to these events.",
        f"Documents show {wrong_answer} was instrumental in the outcome.",
    ]
    
    return random.choice(templates)

def generate_adversarial_squad(output_path="./data/squad_adversarial_strong.json",
                               max_examples=None):
    """
    Generate STRONG adversarial version of SQuAD dev set
    """
    print("Loading SQuAD dev set...")
    dataset = load_dataset("squad", split="validation")
    
    if max_examples:
        dataset = dataset.select(range(max_examples))
    
    print(f"Processing {len(dataset)} examples...")
    
    # Group by context
    context_groups = {}
    for example in tqdm(dataset):
        context = example['context']
        if context not in context_groups:
            context_groups[context] = []
        context_groups[context].append(example)
    
    print("Generating STRONG adversarial examples...")
    
    # Build SQuAD format output
    squad_data = {
        "version": "adversarial-strong-v1.0",
        "data": []
    }
    
    article_id = 0
    for context, examples in tqdm(context_groups.items()):
        # Pick the first example to determine answer type
        first_example = examples[0]
        answer_text = first_example['answers']['text'][0] if first_example['answers']['text'] else ""
        question = first_example['question']
        
        # Determine answer type
        answer_type = extract_answer_type(question, answer_text)
        
        # Create adversarial sentence with wrong but plausible answer
        distractor = create_adversarial_sentence(question, answer_text, answer_type)
        
        # Add distractor at the END of context (this is key - models often look at the end)
        adversarial_context = context + " " + distractor
        
        # Create QAs for this context
        qas = []
        for ex in examples:
            qa = {
                "id": ex['id'],
                "question": ex['question'],
                "answers": ex['answers']
            }
            qas.append(qa)
        
        # Create article structure
        article = {
            "title": f"adversarial_article_{article_id}",
            "paragraphs": [{
                "context": adversarial_context,
                "qas": qas
            }]
        }
        squad_data["data"].append(article)
        article_id += 1
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(squad_data, f, indent=2)
    
    # Print statistics
    num_articles = len(squad_data['data'])
    num_paragraphs = sum(len(article['paragraphs']) for article in squad_data['data'])
    num_qas = sum(
        len(para['qas']) 
        for article in squad_data['data'] 
        for para in article['paragraphs']
    )
    
    print(f"\n✅ Generated STRONG adversarial dataset: {output_path}")
    print(f"\nStatistics:")
    print(f"  Articles: {num_articles}")
    print(f"  Paragraphs: {num_paragraphs}")
    print(f"  Questions: {num_qas}")
    print(f"\nEach example has a STRONG adversarial distractor with plausible wrong answers.")
    print(f"\nUse with: --dataset {output_path}")

if __name__ == "__main__":
    generate_adversarial_squad()
