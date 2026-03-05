"""
Prepare HTAN v1.2.0 benchmarking tasks with simplified prompts and enhanced scoring.

This script:
1. Iterates through HTAN v1.2.0 synthetic datasets
2. Creates task directories in curator-benchmarking/tasks/htan/
3. Copies input_data.tsv and ground_truth.tsv
4. Fetches or copies JSON schemas
5. Generates simple prompts: "Your task: Correct errors in <Schema> metadata records."
6. Copies format_prompt.py template
7. Generates enhanced score.py with F1/precision/recall/confidence metrics
"""

import json
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Any

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SYNTHETIC_DATA_ROOT = REPO_ROOT / "benchmarking/sim-input/synthetic-data/htan2/v1.2.0"
TASKS_ROOT = REPO_ROOT / "curator-benchmarking/tasks/htan"
SCRIPTS_DIR = Path(__file__).parent

# Schema cache to avoid repeated downloads
_schema_cache: Dict[str, Dict[str, Any]] = {}


def fetch_schema(uri: str) -> Dict[str, Any]:
    """Fetch JSON schema from URI with caching."""
    if uri in _schema_cache:
        return _schema_cache[uri]

    try:
        with urllib.request.urlopen(uri) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status} fetching schema from {uri}")
            schema = json.load(response)
    except Exception as exc:
        raise Exception(f"Failed to fetch schema from {uri}: {exc}")

    _schema_cache[uri] = schema
    return schema


def create_simple_prompt(schema_name: str) -> str:
    """Generate simple prompt: Your task: Correct errors in <Schema> metadata records."""
    # Convert snake_case to Title Case
    schema_display = schema_name.replace("_", " ").title()
    return f"Your task: Correct errors in {schema_display} metadata records."


def generate_enhanced_scorer() -> str:
    """Generate enhanced score.py with F1/precision/recall/confidence metrics."""
    return '''"""Enhanced scorer for HTAN data correction with F1/precision/recall/confidence."""
import json
import re
from typing import Dict, Any, Optional


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    # Remove markdown code blocks
    text = re.sub(r'```json\\s*\\n?', '', text)
    text = re.sub(r'```\\s*\\n?', '', text)
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
'''


def get_format_prompt_template() -> str:
    """Get format_prompt.py template from existing tasks."""
    # Try to find an existing format_prompt.py to copy
    existing_tasks = REPO_ROOT / "curator-benchmarking/tasks"

    # Check correction_of_typos first
    typo_task = existing_tasks / "correction_of_typos/format_prompt.py"
    if typo_task.exists():
        return typo_task.read_text()

    # Fallback: generate simple format_prompt.py
    return '''"""Custom prompt formatter for HTAN correction tasks."""
import json
from typing import Dict, Any, Optional


def format_prompt(
    prompt_template: str,
    sample: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    schema: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format prompt with schema information for HTAN data correction.

    Includes:
    - Base prompt template
    - Simplified schema showing enum values and field types
    - Input data as JSON
    """
    # Build simplified schema showing key validation rules
    schema_text = ""
    if schema and "properties" in schema:
        simplified_schema = {
            "type": "object",
            "properties": {}
        }

        properties = schema["properties"]
        for prop_name, prop_def in properties.items():
            field_info = {
                "type": prop_def.get("type", "string")
            }

            # Add description (truncated)
            if "description" in prop_def:
                desc = prop_def["description"]
                field_info["description"] = desc[:100] + "..." if len(desc) > 100 else desc

            # Include enum values (limit to 20 if very large)
            if "enum" in prop_def:
                enum_values = prop_def["enum"]
                if len(enum_values) > 20:
                    field_info["enum_preview"] = enum_values[:20]
                    field_info["enum_count"] = len(enum_values)
                    field_info["enum_note"] = f"Controlled vocabulary with {len(enum_values)} values. First 20 shown."
                else:
                    field_info["enum"] = enum_values

            # Include pattern for ID validation
            if "pattern" in prop_def:
                field_info["pattern"] = prop_def["pattern"]

            # Include range constraints
            if "minimum" in prop_def:
                field_info["minimum"] = prop_def["minimum"]
            if "maximum" in prop_def:
                field_info["maximum"] = prop_def["maximum"]

            # Include array item constraints
            if "items" in prop_def and prop_def.get("type") == "array":
                items = prop_def["items"]
                if "enum" in items:
                    enum_values = items["enum"]
                    if len(enum_values) > 20:
                        field_info["items_enum_preview"] = enum_values[:20]
                        field_info["items_enum_count"] = len(enum_values)
                    else:
                        field_info["items_enum"] = enum_values

            simplified_schema["properties"][prop_name] = field_info

        # Add required fields info
        if "required" in schema:
            simplified_schema["required"] = schema["required"]

        schema_text = f"\\n\\nTarget Schema:\\n{json.dumps(simplified_schema, indent=2)}"

    # Format input data
    sample_text = f"\\n\\nInput Data (with errors to correct):\\n{json.dumps(sample, indent=2)}"

    return f"{prompt_template}{schema_text}{sample_text}"
'''


def create_task(dataset_dir: Path, task_name: str):
    """Create a single HTAN benchmarking task."""
    print(f"\nProcessing {task_name}...")

    # Create task directory
    task_dir = TASKS_ROOT / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created task directory: {task_dir}")

    # Load metadata
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"  WARNING: metadata.json not found in {dataset_dir}")
        metadata = {}
    else:
        metadata = json.loads(metadata_path.read_text())

    # Copy input_data.tsv and ground_truth.tsv
    for filename in ["input_data.tsv", "ground_truth.tsv"]:
        src = dataset_dir / filename
        dst = task_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {filename}")
        else:
            print(f"  WARNING: {filename} not found in {dataset_dir}")

    # Fetch or copy schema
    schema_uri = metadata.get("schemaUri")
    if not schema_uri:
        print(f"  WARNING: schemaUri not found in metadata")
        schema = {}
    else:
        # Check if schema.json exists locally
        local_schema_path = dataset_dir / "schema.json"
        if local_schema_path.exists():
            print(f"  Found local schema.json, copying...")
            shutil.copy2(local_schema_path, task_dir / "schema.json")
            schema = json.loads(local_schema_path.read_text())
        else:
            print(f"  Fetching schema from {schema_uri}...")
            schema = fetch_schema(schema_uri)
            # Save to task directory
            (task_dir / "schema.json").write_text(json.dumps(schema, indent=2))
            print(f"  Saved schema.json")

    # Generate simple prompt
    schema_name = task_name.replace("htan_", "")
    prompt_content = create_simple_prompt(schema_name)
    (task_dir / "default_prompt.txt").write_text(prompt_content)
    print(f"  Generated default_prompt.txt: \"{prompt_content}\"")

    # Generate format_prompt.py
    format_prompt_content = get_format_prompt_template()
    (task_dir / "format_prompt.py").write_text(format_prompt_content)
    print(f"  Generated format_prompt.py")

    # Generate enhanced score.py
    score_content = generate_enhanced_scorer()
    (task_dir / "score.py").write_text(score_content)
    print(f"  Generated enhanced score.py")

    print(f"  ✓ Task {task_name} created successfully")


def main():
    """Main function to prepare all HTAN tasks."""
    print("=" * 70)
    print("HTAN v1.2.0 Task Preparation (Simplified with Enhanced Scoring)")
    print("=" * 70)

    if not SYNTHETIC_DATA_ROOT.exists():
        print(f"ERROR: Synthetic data root not found: {SYNTHETIC_DATA_ROOT}")
        return

    # Get all dataset directories
    dataset_dirs = [d for d in SYNTHETIC_DATA_ROOT.iterdir() if d.is_dir()]
    dataset_dirs.sort()

    print(f"\nFound {len(dataset_dirs)} datasets:")
    for d in dataset_dirs:
        print(f"  - {d.name}")

    # Create tasks
    print("\nCreating tasks...")
    for dataset_dir in dataset_dirs:
        schema_type = dataset_dir.name
        task_name = f"htan_{schema_type}"
        try:
            create_task(dataset_dir, task_name)
        except Exception as e:
            print(f"  ERROR creating task {task_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Task preparation complete!")
    print("=" * 70)
    print(f"\nTasks created in: {TASKS_ROOT}")
    print("\nNext steps:")
    print("  1. cd curator-benchmarking")
    print("  2. python -m src.cli list | grep htan")
    print("  3. python -m src.cli run htan/htan_demographics --limit 5")


if __name__ == "__main__":
    main()
