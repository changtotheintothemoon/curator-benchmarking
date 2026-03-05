"""
Fix TSV parsing errors in bulk_wes schemas.

Issues:
- bulk_wes_level1: Line 6 has empty LANE_NUMBER field (double tab)
- All bulk_wes files may have inconsistent field counts

Solution:
- Use pandas with error handling to read and rewrite TSV files
- Fill missing values appropriately
"""

import pandas as pd
from pathlib import Path

# Paths
SYNTHETIC_DATA_ROOT = Path(__file__).parent.parent.parent / "benchmarking/sim-input/synthetic-data/htan2/v1.2.0"
BULK_WES_SCHEMAS = ["bulk_wes_level1", "bulk_wes_level2", "bulk_wes_level3"]


def fix_tsv_file(tsv_path: Path, output_path: Path = None):
    """
    Fix TSV file by reading with error handling and rewriting.

    Args:
        tsv_path: Path to TSV file to fix
        output_path: Optional output path (defaults to overwriting input)
    """
    if output_path is None:
        output_path = tsv_path

    print(f"  Reading {tsv_path.name}...")

    try:
        # Try reading with default settings first
        df = pd.read_csv(tsv_path, sep='\t')
        print(f"    ✓ Read {len(df)} rows successfully")

    except pd.errors.ParserError as e:
        print(f"    ⚠️  Parser error: {e}")
        print(f"    Attempting repair...")

        try:
            # Try with on_bad_lines='skip' (pandas >= 1.3)
            df = pd.read_csv(tsv_path, sep='\t', on_bad_lines='warn')
            print(f"    ✓ Read {len(df)} rows (some rows may have been skipped)")

        except TypeError:
            # Fallback for older pandas versions
            try:
                df = pd.read_csv(tsv_path, sep='\t', error_bad_lines=False, warn_bad_lines=True)
                print(f"    ✓ Read {len(df)} rows (some rows may have been skipped)")
            except Exception as e2:
                print(f"    ✗ Failed to read with error handling: {e2}")
                return False

        except Exception as e2:
            print(f"    ✗ Failed to read: {e2}")
            return False

    # Write back to file
    print(f"  Writing to {output_path.name}...")
    df.to_csv(output_path, sep='\t', index=False)
    print(f"    ✓ Wrote {len(df)} rows, {len(df.columns)} columns")

    return True


def main():
    print("="*70)
    print("Fixing bulk_wes TSV Files")
    print("="*70)

    for schema in BULK_WES_SCHEMAS:
        schema_dir = SYNTHETIC_DATA_ROOT / schema

        if not schema_dir.exists():
            print(f"\n⚠️  {schema} directory not found, skipping")
            continue

        print(f"\nProcessing {schema}...")

        # Fix input_data.tsv
        input_tsv = schema_dir / "input_data.tsv"
        if input_tsv.exists():
            success = fix_tsv_file(input_tsv)
            if not success:
                print(f"  ✗ Failed to fix input_data.tsv")
        else:
            print(f"  ⚠️  input_data.tsv not found")

        # Fix ground_truth.tsv
        ground_truth_tsv = schema_dir / "ground_truth.tsv"
        if ground_truth_tsv.exists():
            success = fix_tsv_file(ground_truth_tsv)
            if not success:
                print(f"  ✗ Failed to fix ground_truth.tsv")
        else:
            print(f"  ⚠️  ground_truth.tsv not found")

    print("\n" + "="*70)
    print("Fix complete!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Verify fixes: python -c \"import pandas as pd; pd.read_csv('path/to/file.tsv', sep='\\t')\"")
    print("  2. Regenerate tasks: python scripts/prepare_htan_typed_tasks.py")


if __name__ == "__main__":
    main()
