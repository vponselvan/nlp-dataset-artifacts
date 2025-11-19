#!/usr/bin/env python3
"""
Linguistic and Adversarial Pattern Analysis for AddSent
Categorizes errors by specific adversarial techniques and linguistic patterns
"""

import json
import re
from collections import defaultdict
from typing import Dict, List, Any

def load_data(predictions_path: str, dataset_path: str) -> tuple:
    """Load predictions and dataset"""
    with open(predictions_path, 'r') as f:
        predictions = [json.loads(line) for line in f]
    
    with open(dataset_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    return predictions, dataset

def detect_entity_substitution(context: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect if model picked a substitute entity of the same type
    e.g., predicted "Chicago" when answer is "Santa Clara" (both cities)
    """
    gt_lower = ground_truth.lower().strip()
    pred_lower = prediction.lower().strip()
    
    # Both in context and different
    if gt_lower in context.lower() and pred_lower in context.lower() and gt_lower != pred_lower:
        # Check if they're proper nouns (capitalized entities)
        gt_words = ground_truth.split()
        pred_words = prediction.split()
        if gt_words and pred_words:
            if gt_words[0][0].isupper() and pred_words[0][0].isupper():
                return True
    return False

def detect_negation_confusion(context: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect if error involves negation (not, never, no, didn't, etc.)
    """
    negation_words = ['not', 'never', 'no', "n't", 'neither', 'nor', 'nobody', 'nothing', 'none']
    
    # Find sentences containing ground truth and prediction
    sentences = context.split('.')
    gt_sent = [s for s in sentences if ground_truth.lower() in s.lower()]
    pred_sent = [s for s in sentences if prediction.lower() in s.lower()]
    
    # Check if negation appears near either
    for sent in gt_sent:
        if any(neg in sent.lower() for neg in negation_words):
            return True
    for sent in pred_sent:
        if any(neg in sent.lower() for neg in negation_words):
            return True
    
    return False

def detect_temporal_confusion(ground_truth: str, prediction: str) -> bool:
    """
    Detect if error involves temporal expressions (dates, years, times)
    """
    # Both contain years
    gt_has_year = bool(re.search(r'\b\d{4}\b', ground_truth))
    pred_has_year = bool(re.search(r'\b\d{4}\b', prediction))
    
    # Both contain dates
    months = ['january', 'february', 'march', 'april', 'may', 'june', 
              'july', 'august', 'september', 'october', 'november', 'december']
    gt_has_date = any(month in ground_truth.lower() for month in months)
    pred_has_date = any(month in prediction.lower() for month in months)
    
    return (gt_has_year and pred_has_year) or (gt_has_date and pred_has_date)

def detect_numeric_confusion(ground_truth: str, prediction: str) -> bool:
    """
    Detect if error involves numbers (scores, quantities, counts)
    """
    gt_has_num = bool(re.search(r'\d+', ground_truth))
    pred_has_num = bool(re.search(r'\d+', prediction))
    
    return gt_has_num and pred_has_num and ground_truth.strip() != prediction.strip()

def detect_paraphrase_distractor(context: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect if prediction is a paraphrase/synonym of ground truth
    """
    gt_lower = ground_truth.lower().strip()
    pred_lower = prediction.lower().strip()
    
    # Partial overlap in words
    gt_words = set(gt_lower.split())
    pred_words = set(pred_lower.split())
    
    if gt_words and pred_words:
        overlap = len(gt_words.intersection(pred_words))
        if overlap > 0 and overlap < min(len(gt_words), len(pred_words)):
            return True
    
    # Common paraphrase patterns
    paraphrase_pairs = [
        ('won', 'victory'), ('defeated', 'beat'), ('champion', 'winner'),
        ('located', 'situated'), ('named', 'called'), ('began', 'started')
    ]
    
    for p1, p2 in paraphrase_pairs:
        if (p1 in gt_lower and p2 in pred_lower) or (p2 in gt_lower and p1 in pred_lower):
            return True
    
    return False

def detect_coreference_error(context: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect if error involves pronouns or coreference (he, she, it, they, this, that)
    """
    pronouns = ['he', 'she', 'it', 'they', 'them', 'his', 'her', 'their', 
                'this', 'that', 'these', 'those', 'who', 'which']
    
    pred_lower = prediction.lower().strip()
    
    # Prediction is a pronoun but answer is not
    if pred_lower in pronouns and ground_truth.lower().strip() not in pronouns:
        return True
    
    return False

def detect_modal_confusion(context: str, ground_truth: str) -> bool:
    """
    Detect if context contains modal verbs (might, could, should, would, may)
    near the ground truth - indicates uncertainty/conditionality
    """
    modals = ['might', 'could', 'should', 'would', 'may', 'possibly', 
              'perhaps', 'probably', 'likely', 'unlikely']
    
    # Find sentence containing ground truth
    sentences = context.split('.')
    for sent in sentences:
        if ground_truth.lower() in sent.lower():
            if any(modal in sent.lower() for modal in modals):
                return True
    
    return False

def detect_comparative_superlative_error(question: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect errors in comparative/superlative questions (most, least, best, worst, more, less)
    """
    comparatives = ['more', 'less', 'better', 'worse', 'higher', 'lower', 
                    'greater', 'fewer', 'longer', 'shorter']
    superlatives = ['most', 'least', 'best', 'worst', 'highest', 'lowest', 
                    'greatest', 'longest', 'shortest', 'first', 'last']
    
    question_lower = question.lower()
    
    has_comparative = any(comp in question_lower for comp in comparatives)
    has_superlative = any(sup in question_lower for sup in superlatives)
    
    return has_comparative or has_superlative

def detect_additive_sentence_pattern(context: str) -> bool:
    """
    Detect if context contains AddSent-specific patterns:
    - "However, ..."
    - "According to ..."
    - "Some sources say ..."
    - "It is also known that ..."
    """
    additive_patterns = [
        r'\bhowever\b', r'\baccording to\b', r'\bsome sources\b',
        r'\bit is (also )?known that\b', r'\bin contrast\b', r'\bon the other hand\b',
        r'\bnevertheless\b', r'\bfurthermore\b', r'\bmoreover\b',
        r'\badditionally\b', r'\balternatively\b'
    ]
    
    context_lower = context.lower()
    for pattern in additive_patterns:
        if re.search(pattern, context_lower):
            return True
    
    return False

def detect_list_enumeration_error(context: str, ground_truth: str, prediction: str) -> bool:
    """
    Detect errors in list/enumeration contexts (and, or, comma-separated items)
    """
    # Check if ground truth and prediction are both in a list context
    gt_pattern = re.escape(ground_truth.lower())
    pred_pattern = re.escape(prediction.lower())
    
    # Look for comma-separated lists
    list_pattern_gt = f'({gt_pattern}[,;]|[,;]\\s*{gt_pattern})'
    list_pattern_pred = f'({pred_pattern}[,;]|[,;]\\s*{pred_pattern})'
    
    in_list_gt = bool(re.search(list_pattern_gt, context.lower()))
    in_list_pred = bool(re.search(list_pattern_pred, context.lower()))
    
    return in_list_gt and in_list_pred

def analyze_linguistic_patterns(predictions: List[Dict], dataset: List[Dict]) -> Dict:
    """Analyze errors by linguistic and adversarial patterns"""
    
    pred_dict = {p['id']: p for p in predictions}
    
    results = {
        "summary": {"total": 0, "correct": 0, "incorrect": 0},
        "adversarial_patterns": {
            "entity_substitution": {"count": 0, "examples": []},
            "negation_confusion": {"count": 0, "examples": []},
            "temporal_confusion": {"count": 0, "examples": []},
            "numeric_confusion": {"count": 0, "examples": []},
            "paraphrase_distractor": {"count": 0, "examples": []},
            "coreference_error": {"count": 0, "examples": []},
            "modal_confusion": {"count": 0, "examples": []},
            "comparative_superlative": {"count": 0, "examples": []},
            "additive_sentence": {"count": 0, "examples": []},
            "list_enumeration": {"count": 0, "examples": []},
        },
        "pattern_combinations": defaultdict(int),
        "most_common_combinations": []
    }
    
    for example in dataset:
        example_id = example['id']
        if example_id not in pred_dict:
            continue
        
        pred = pred_dict[example_id]
        ground_truth = example['answers']['text'][0]
        prediction = pred.get('predicted_answer', pred.get('prediction_text', ''))
        context = example['context']
        question = example['question']
        
        # Check correctness
        is_correct = ground_truth.lower().strip() == prediction.lower().strip()
        
        results["summary"]["total"] += 1
        if is_correct:
            results["summary"]["correct"] += 1
        else:
            results["summary"]["incorrect"] += 1
            
            # Only analyze patterns for incorrect predictions
            patterns_detected = []
            
            # Detect each pattern
            if detect_entity_substitution(context, ground_truth, prediction):
                results["adversarial_patterns"]["entity_substitution"]["count"] += 1
                patterns_detected.append("entity_substitution")
                if len(results["adversarial_patterns"]["entity_substitution"]["examples"]) < 3:
                    results["adversarial_patterns"]["entity_substitution"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_negation_confusion(context, ground_truth, prediction):
                results["adversarial_patterns"]["negation_confusion"]["count"] += 1
                patterns_detected.append("negation_confusion")
                if len(results["adversarial_patterns"]["negation_confusion"]["examples"]) < 3:
                    results["adversarial_patterns"]["negation_confusion"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_temporal_confusion(ground_truth, prediction):
                results["adversarial_patterns"]["temporal_confusion"]["count"] += 1
                patterns_detected.append("temporal_confusion")
                if len(results["adversarial_patterns"]["temporal_confusion"]["examples"]) < 3:
                    results["adversarial_patterns"]["temporal_confusion"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_numeric_confusion(ground_truth, prediction):
                results["adversarial_patterns"]["numeric_confusion"]["count"] += 1
                patterns_detected.append("numeric_confusion")
                if len(results["adversarial_patterns"]["numeric_confusion"]["examples"]) < 3:
                    results["adversarial_patterns"]["numeric_confusion"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_paraphrase_distractor(context, ground_truth, prediction):
                results["adversarial_patterns"]["paraphrase_distractor"]["count"] += 1
                patterns_detected.append("paraphrase_distractor")
                if len(results["adversarial_patterns"]["paraphrase_distractor"]["examples"]) < 3:
                    results["adversarial_patterns"]["paraphrase_distractor"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_coreference_error(context, ground_truth, prediction):
                results["adversarial_patterns"]["coreference_error"]["count"] += 1
                patterns_detected.append("coreference_error")
                if len(results["adversarial_patterns"]["coreference_error"]["examples"]) < 3:
                    results["adversarial_patterns"]["coreference_error"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_modal_confusion(context, ground_truth):
                results["adversarial_patterns"]["modal_confusion"]["count"] += 1
                patterns_detected.append("modal_confusion")
                if len(results["adversarial_patterns"]["modal_confusion"]["examples"]) < 3:
                    results["adversarial_patterns"]["modal_confusion"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_comparative_superlative_error(question, ground_truth, prediction):
                results["adversarial_patterns"]["comparative_superlative"]["count"] += 1
                patterns_detected.append("comparative_superlative")
                if len(results["adversarial_patterns"]["comparative_superlative"]["examples"]) < 3:
                    results["adversarial_patterns"]["comparative_superlative"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_additive_sentence_pattern(context):
                results["adversarial_patterns"]["additive_sentence"]["count"] += 1
                patterns_detected.append("additive_sentence")
                if len(results["adversarial_patterns"]["additive_sentence"]["examples"]) < 3:
                    results["adversarial_patterns"]["additive_sentence"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            if detect_list_enumeration_error(context, ground_truth, prediction):
                results["adversarial_patterns"]["list_enumeration"]["count"] += 1
                patterns_detected.append("list_enumeration")
                if len(results["adversarial_patterns"]["list_enumeration"]["examples"]) < 3:
                    results["adversarial_patterns"]["list_enumeration"]["examples"].append({
                        "id": example_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "prediction": prediction
                    })
            
            # Track pattern combinations
            if patterns_detected:
                pattern_combo = "+".join(sorted(patterns_detected))
                results["pattern_combinations"][pattern_combo] += 1
    
    # Get top 20 most common pattern combinations
    sorted_combos = sorted(results["pattern_combinations"].items(), 
                          key=lambda x: x[1], reverse=True)[:20]
    results["most_common_combinations"] = [
        {"pattern": combo, "count": count} 
        for combo, count in sorted_combos
    ]
    
    # Convert defaultdict to dict
    results["pattern_combinations"] = dict(results["pattern_combinations"])
    
    return results

def main():
    predictions_path = "./eval_results_adversarial/eval_predictions.jsonl"
    dataset_path = "./data/addsent_eval.jsonl"
    output_path = "./analysis/linguistic_pattern_analysis.json"
    
    print("Loading data...")
    predictions, dataset = load_data(predictions_path, dataset_path)
    print(f"Loaded {len(predictions)} predictions and {len(dataset)} examples")
    
    print("\nAnalyzing linguistic and adversarial patterns...")
    results = analyze_linguistic_patterns(predictions, dataset)
    
    print(f"\nTotal: {results['summary']['total']}")
    print(f"Correct: {results['summary']['correct']}")
    print(f"Incorrect: {results['summary']['incorrect']}")
    
    print("\n=== Adversarial Pattern Detection ===")
    sorted_patterns = sorted(results['adversarial_patterns'].items(), 
                            key=lambda x: x[1]['count'], reverse=True)
    
    for pattern_name, data in sorted_patterns:
        count = data['count']
        percentage = (count / results['summary']['incorrect'] * 100) if results['summary']['incorrect'] > 0 else 0
        print(f"{pattern_name:25s}: {count:4d} errors ({percentage:5.1f}% of incorrect)")
    
    print("\n=== Top Pattern Combinations ===")
    for combo_data in results['most_common_combinations'][:10]:
        pattern = combo_data['pattern']
        count = combo_data['count']
        percentage = (count / results['summary']['incorrect'] * 100) if results['summary']['incorrect'] > 0 else 0
        print(f"{pattern:50s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")

if __name__ == "__main__":
    main()
