"""
Prepare HTAN v1.2.0 benchmarking tasks with three task types.

Creates 39 tasks (13 schemas × 3 types):
1. Typo correction - Fix misspellings, case errors, formatting issues
2. Synonym narrowing - Replace broad terms with specific controlled vocabulary
3. Value imputation - Fill missing/null values based on patterns

Each task gets:
- Appropriate synthetic data (filtered/generated from v1.2.0)
- Task-specific detailed prompt
- Enhanced scorer with F1/precision/recall/confidence
- Schema and format_prompt.py
"""

import json
import shutil
import urllib.request
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Paths
REPO_ROOT = Path(__file__).parent.parent.parent
SYNTHETIC_DATA_ROOT = REPO_ROOT / "benchmarking/sim-input/synthetic-data/htan2/v1.2.0"
TASKS_ROOT = REPO_ROOT / "curator-benchmarking/tasks"
SCRIPTS_DIR = Path(__file__).parent

# Schema cache
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


def generate_typo_prompt(schema_name: str, metadata: Dict[str, Any]) -> str:
    """Generate detailed prompt for typo correction tasks."""
    complexity = metadata.get("complexity", "medium")
    error_notes = metadata.get("coverageNotes", "Various error types present")

    schema_display = schema_name.replace("_", " ").title()

    return f'''You are a metadata curation assistant specializing in HTAN biomedical data quality.

Your task: Correct errors in {schema_display} metadata records.

Dataset Complexity: {complexity}
Common error types in the input:
{error_notes[:500]}

General error types to watch for:
- Enum values with wrong case (e.g., "male" should be "Male")
- Invalid enum values not in controlled vocabulary
- Misspellings and typos in text values
- Invalid ID formats (HTAN IDs must match specific patterns)
- Out-of-range numeric values (negative ages, percentages > 100)
- Leading/trailing whitespace
- Wrong array separators (semicolons instead of commas, or vice versa)
- Conditional validation failures (e.g., "Other" selected without specification)

INSTRUCTIONS:
1. Carefully review the input record and identify ALL errors
2. Correct each error by:
   - Matching exact enum values from the Target Schema (case-sensitive)
   - Fixing misspellings and typos
   - Fixing ID formats to match HTAN patterns
   - Ensuring numeric values are within valid ranges
   - Removing extra whitespace
   - Using proper separators for array values (check schema)
3. For free-text fields (*_OTHER_SPECIFY), preserve meaning while fixing formatting
4. Return ONLY the corrected record as valid JSON
5. Preserve all original fields even if they don't need correction

CRITICAL RULES:
- Use EXACT enum values from the schema (case-sensitive matching required)
- Do not add fields that weren't in the input
- Do not remove fields unless they're completely invalid
- Ensure correct data types (integer vs number vs string vs array)
- For arrays, use the proper separator (usually commas between items)

Output Format:
```json
{{
  "FIELD_1": "corrected_value",
  "FIELD_2": 123,
  "ARRAY_FIELD": ["value1", "value2"],
  ...
}}
```

Return ONLY the JSON. No explanation needed.'''


def generate_synonym_prompt(schema_name: str, metadata: Dict[str, Any]) -> str:
    """Generate detailed prompt for synonym narrowing tasks."""
    complexity = metadata.get("complexity", "medium")

    schema_display = schema_name.replace("_", " ").title()

    return f'''You are a metadata curation assistant specializing in HTAN biomedical data quality.

Your task: Narrow broad synonyms in {schema_display} metadata records.

Dataset Complexity: {complexity}

The input data contains metadata entries with broad, general terms that should be narrowed to more specific, precise terms. Please correct each entry by replacing broad synonyms with their narrower, more specific equivalents.

Common synonym patterns to narrow:
- Broad disease names → Specific disease classifications
- General terms → Precise medical terminology
- Abbreviations → Full standard terms (e.g., "M" → "Male", "NF" → "Neurofibromatosis type 1")
- Colloquial terms → Controlled vocabulary terms
- Partial matches → Complete enum values (e.g., "Hispanic" → "Hispanic or Latino")
- Common synonyms → Standard terms (e.g., "Caucasian" → "White")

IMPORTANT: When narrowing values, you must use the exact controlled terminology values from the Target Schema provided below. For fields with enumerated values (enums), you must match one of the exact values from the schema. Do not use variations or synonyms - use the exact values specified in the schema.

INSTRUCTIONS:
1. Identify fields with broad or synonymous terms
2. Replace with more specific terms from the Target Schema
3. Ensure the narrowed term is contextually appropriate
4. Preserve all original fields
5. Return ONLY the corrected record as valid JSON

CRITICAL RULES:
- Use EXACT enum values from the schema (case-sensitive matching required)
- Narrow to the most specific appropriate term
- Do not add fields that weren't in the input
- Maintain data integrity and logical consistency

Output Format:
```json
{{
  "FIELD_1": "narrowed_specific_value",
  "FIELD_2": "controlled_terminology_value",
  ...
}}
```

Return ONLY the JSON. No explanation needed.'''


def generate_imputation_prompt(schema_name: str, metadata: Dict[str, Any]) -> str:
    """Generate detailed prompt for value imputation tasks."""
    complexity = metadata.get("complexity", "medium")

    schema_display = schema_name.replace("_", " ").title()

    return f'''You are a metadata curation assistant specializing in HTAN biomedical data quality.

Your task: Impute (fill in) missing values in {schema_display} metadata records.

Dataset Complexity: {complexity}

The input data contains metadata entries with missing or null values. Please fill in these missing values based on:
1. Patterns observed in other rows/columns
2. Required fields defined in the schema
3. Conditional field requirements (e.g., if X="Other", fill X_OTHER_SPECIFY)
4. Logical consistency with other field values

IMPORTANT: When imputing values, you must use the exact controlled terminology values from the Target Schema provided below. For fields with enumerated values (enums), you must match one of the exact values from the schema.

Imputation strategies:
- Use mode/most common value for categorical fields when appropriate
- Infer from conditional relationships in the schema
- Use schema defaults where defined
- Maintain logical consistency (temporal ordering, anatomical relationships)
- Fill required fields when context provides sufficient information

INSTRUCTIONS:
1. Identify missing/null values in the record
2. Determine appropriate imputation strategy based on:
   - Field type (required vs optional)
   - Schema constraints (enums, patterns, ranges)
   - Context from other fields in the same record
3. Fill with appropriate values from Target Schema
4. Maintain data integrity and logical consistency
5. Return ONLY the complete record as valid JSON

CRITICAL RULES:
- Use EXACT enum values from the schema (case-sensitive matching required)
- Only impute when there is sufficient context or clear requirement
- Do not add fields that weren't in the input
- Preserve all original non-missing fields unchanged
- Ensure correct data types (integer vs number vs string vs array)

Output Format:
```json
{{
  "FIELD_1": "imputed_value",
  "FIELD_2": "original_value",
  "PREVIOUSLY_NULL_FIELD": "newly_imputed_value",
  ...
}}
```

Return ONLY the JSON. No explanation needed.'''


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
    """Get format_prompt.py template."""
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


def filter_typo_records(input_df: pd.DataFrame, ground_truth_df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:
    """Filter records that have typo/case/formatting errors."""
    # For now, use first 15 records that have errors
    # In future, could analyze error types from metadata to be more selective
    error_breakdown = metadata.get("errorTypeBreakdown", {})
    typo_count = error_breakdown.get("typos", 0)
    case_count = error_breakdown.get("caseSensitivity", 0)

    # Take up to 15 records with errors (skip valid records)
    error_count = metadata.get("errorCount", len(input_df))
    valid_count = metadata.get("validCount", 0)

    # Skip valid records at the end, take error records
    max_records = min(15, error_count)

    return input_df.head(max_records), ground_truth_df.head(max_records)


def filter_synonym_records(input_df: pd.DataFrame, ground_truth_df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:
    """Filter/generate records with synonym issues."""
    # For demographics, use records with synonym errors
    # For others, generate synonym variants by broadening ground truth terms

    error_breakdown = metadata.get("errorTypeBreakdown", {})
    synonym_count = error_breakdown.get("synonyms", 0)
    abbrev_count = error_breakdown.get("abbreviations", 0)

    if synonym_count > 0 or abbrev_count > 0:
        # Use existing synonym records
        max_records = min(12, synonym_count + abbrev_count)
        return input_df.head(max_records), ground_truth_df.head(max_records)
    else:
        # Generate synonym variants (placeholder - will need enhancement)
        # For now, use subset of records
        max_records = min(10, len(input_df))
        return input_df.head(max_records), ground_truth_df.head(max_records)


def filter_imputation_records(input_df: pd.DataFrame, ground_truth_df: pd.DataFrame, metadata: Dict[str, Any]) -> tuple:
    """Filter/generate records with missing values."""
    error_breakdown = metadata.get("errorTypeBreakdown", {})
    missing_count = error_breakdown.get("missingValues", 0)

    if missing_count > 0:
        # Use existing records with missing values
        max_records = min(12, missing_count)
        return input_df.head(max_records), ground_truth_df.head(max_records)
    else:
        # Generate records with missing values by nulling some fields
        # For now, use subset of records
        max_records = min(10, len(input_df))
        return input_df.head(max_records), ground_truth_df.head(max_records)


def create_typed_task(dataset_dir: Path, schema_name: str, task_type: str):
    """Create a single typed HTAN benchmarking task."""
    task_name = f"htan_{schema_name}_{task_type}"
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

    # Load source data
    input_df = pd.read_csv(dataset_dir / "input_data.tsv", sep='\t')
    ground_truth_df = pd.read_csv(dataset_dir / "ground_truth.tsv", sep='\t')

    # Filter records based on task type
    if task_type == "typos":
        filtered_input, filtered_ground_truth = filter_typo_records(input_df, ground_truth_df, metadata)
    elif task_type == "synonyms":
        filtered_input, filtered_ground_truth = filter_synonym_records(input_df, ground_truth_df, metadata)
    elif task_type == "imputation":
        filtered_input, filtered_ground_truth = filter_imputation_records(input_df, ground_truth_df, metadata)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Save filtered data
    filtered_input.to_csv(task_dir / "input_data.tsv", sep='\t', index=False)
    filtered_ground_truth.to_csv(task_dir / "ground_truth.tsv", sep='\t', index=False)
    print(f"  Saved {len(filtered_input)} records")

    # Copy or fetch schema
    schema_uri = metadata.get("schemaUri")
    if not schema_uri:
        print(f"  WARNING: schemaUri not found in metadata")
        schema = {}
    else:
        local_schema_path = dataset_dir / "schema.json"
        if local_schema_path.exists():
            print(f"  Found local schema.json, copying...")
            shutil.copy2(local_schema_path, task_dir / "schema.json")
            schema = json.loads(local_schema_path.read_text())
        else:
            print(f"  Fetching schema from {schema_uri}...")
            schema = fetch_schema(schema_uri)
            (task_dir / "schema.json").write_text(json.dumps(schema, indent=2))
            print(f"  Saved schema.json")

    # Generate task-specific prompt
    if task_type == "typos":
        prompt_content = generate_typo_prompt(schema_name, metadata)
    elif task_type == "synonyms":
        prompt_content = generate_synonym_prompt(schema_name, metadata)
    elif task_type == "imputation":
        prompt_content = generate_imputation_prompt(schema_name, metadata)

    (task_dir / "default_prompt.txt").write_text(prompt_content)
    print(f"  Generated {task_type}-specific prompt")

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
    """Main function to prepare all 39 HTAN tasks (13 × 3)."""
    print("=" * 70)
    print("HTAN v1.2.0 Typed Task Preparation (39 tasks: 13 × 3)")
    print("=" * 70)

    if not SYNTHETIC_DATA_ROOT.exists():
        print(f"ERROR: Synthetic data root not found: {SYNTHETIC_DATA_ROOT}")
        return

    # Get all dataset directories
    dataset_dirs = [d for d in SYNTHETIC_DATA_ROOT.iterdir() if d.is_dir()]
    dataset_dirs.sort()

    # NOTE: bulk_wes TSV issues have been fixed by fix_bulk_wes_tsv.py
    # If you encounter TSV parsing errors again, uncomment the following lines:
    # skip_schemas = ["bulk_wes_level1", "bulk_wes_level2", "bulk_wes_level3"]
    # dataset_dirs = [d for d in dataset_dirs if d.name not in skip_schemas]

    print(f"\nFound {len(dataset_dirs)} datasets")
    for d in dataset_dirs:
        print(f"  - {d.name}")

    # Task types
    task_types = ["typos", "synonyms", "imputation"]

    # Create tasks
    print(f"\nCreating {len(dataset_dirs)} × {len(task_types)} = {len(dataset_dirs) * len(task_types)} tasks...")

    for dataset_dir in dataset_dirs:
        schema_name = dataset_dir.name
        for task_type in task_types:
            try:
                create_typed_task(dataset_dir, schema_name, task_type)
            except Exception as e:
                print(f"  ERROR creating task {schema_name}_{task_type}: {e}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 70)
    print("Task preparation complete!")
    print("=" * 70)
    print(f"\nTasks created in: {TASKS_ROOT}")
    print(f"\nTotal tasks: {len(dataset_dirs) * len(task_types)}")
    print("\nNext steps:")
    print("  1. cd curator-benchmarking")
    print("  2. python -m src.cli list | grep htan")
    print("  3. python -m src.cli run htan_demographics_typos --test")


if __name__ == "__main__":
    main()
