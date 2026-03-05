"""Enhanced scorer for HTAN data correction with F1/precision/recall/confidence."""
import json
import re
from typing import Dict, Any, Optional


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*\n?', '', text)
    text = re.sub(r'```\s*\n?', '', text)
    text = text.strip()

    # Try to find JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')

    if start != -1 and end != -1 and end > start:
        return text[start:end+1]

    return text


def score(
    prediction: str,
    ground_truth: Dict[str, Any],
    input_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Enhanced scorer with F1, precision, recall, and confidence metrics.

    Returns dict with:
    - score: Field-level accuracy (correct fields / total fields)
    - f1: Harmonic mean of precision and recall
    - precision: Correct corrections / total changes made
    - recall: Correct corrections / corrections needed
    - confidence: Enum validation rate
    - tp/fp/tn/fn: True/false positives/negatives
    """
    try:
        # Extract and parse JSON
        json_str = _extract_json(prediction)
        if not json_str:
            return {
                "score": 0.0, "f1": 0.0, "precision": 0.0,
                "recall": 0.0, "confidence": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0
            }

        try:
            pred_dict = json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return {
                "score": 0.0, "f1": 0.0, "precision": 0.0,
                "recall": 0.0, "confidence": 0.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0
            }

        # Get schema for enum validation
        schema = input_data.get("_schema", {}) if input_data else {}
        properties = schema.get("properties", {})

        # Calculate metrics
        all_keys = set(pred_dict.keys()) | set(ground_truth.keys())
        if not all_keys:
            return {
                "score": 1.0, "f1": 1.0, "precision": 1.0,
                "recall": 1.0, "confidence": 1.0,
                "tp": 0, "fp": 0, "tn": 0, "fn": 0
            }

        tp = 0  # True positives: field corrected correctly
        fp = 0  # False positives: field changed incorrectly
        tn = 0  # True negatives: correct field left unchanged
        fn = 0  # False negatives: incorrect field not fixed

        correct_fields = 0
        enum_matches = 0
        enum_total = 0

        for key in all_keys:
            pred_val = pred_dict.get(key)
            truth_val = ground_truth.get(key)
            input_val = input_data.get(key) if input_data else None

            # Check if field has enum
            prop_schema = properties.get(key, {})
            if "enum" in prop_schema:
                enum_total += 1
                if pred_val in prop_schema["enum"]:
                    enum_matches += 1

            # Calculate TP/FP/TN/FN
            if pred_val == truth_val:
                correct_fields += 1
                if input_val == truth_val:
                    tn += 1  # Already correct, left unchanged
                else:
                    tp += 1  # Was wrong, corrected successfully
            else:
                if input_val == truth_val:
                    fp += 1  # Was correct, changed incorrectly
                else:
                    fn += 1  # Was wrong, not fixed or fixed incorrectly

        # Calculate final metrics
        total_fields = len(all_keys)
        score_val = correct_fields / total_fields if total_fields > 0 else 0.0

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        confidence = enum_matches / enum_total if enum_total > 0 else 1.0

        return {
            "score": score_val,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "confidence": confidence,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    except Exception as e:
        print(f"Error scoring prediction: {e}")
        return {
            "score": 0.0, "f1": 0.0, "precision": 0.0,
            "recall": 0.0, "confidence": 0.0,
            "tp": 0, "fp": 0, "tn": 0, "fn": 0
        }
