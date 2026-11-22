"""
TechnoHome Customer Support Ticket Router - Data Download & Preparation
Downloads CLINC150 from HuggingFace, maps intents to departments, and merges
with hand-written domain examples.
"""

import json
import os
import sys
from collections import Counter

# Allow importing config from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_domain_examples(path: str) -> list[dict]:
    """Load hand-written bilingual domain examples."""
    with open(path, "r", encoding="utf-8") as f:
        examples = json.load(f)
    print(f"  Loaded {len(examples)} domain examples from {os.path.basename(path)}")
    return examples


def download_clinc150() -> dict:
    """Download the CLINC150 (OOS) dataset from HuggingFace."""
    from datasets import load_dataset

    print("Downloading CLINC150 dataset from HuggingFace...")
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)
    print(f"  Splits available: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")
    return dataset


def get_intent_name(intent_idx: int, dataset) -> str:
    """Convert integer intent index to intent name using dataset features."""
    intent_feature = dataset["train"].features["intent"]
    return intent_feature.int2str(intent_idx)


def map_to_departments(dataset) -> dict:
    """Map CLINC150 examples to TechnoHome departments.

    Returns a dict with train/validation/test splits, each containing
    a list of examples with text, original_intent, department, and source.
    """
    split_mapping = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    prepared = {}
    unmapped_intents = Counter()

    for hf_split, our_split in split_mapping.items():
        examples = []
        split_data = dataset[hf_split]

        for item in split_data:
            text = item["text"]
            intent_idx = item["intent"]
            intent_name = get_intent_name(intent_idx, dataset)

            # Map intent to department
            department = config.INTENT_TO_DEPARTMENT.get(intent_name)
            if department is None:
                unmapped_intents[intent_name] += 1
                continue

            examples.append({
                "text": text,
                "original_intent": intent_name,
                "department": department,
                "lang": "en",
                "source": "clinc150",
            })

        prepared[our_split] = examples
        print(f"  {our_split}: {len(examples)} mapped examples")

    if unmapped_intents:
        print(f"\n  Skipped {sum(unmapped_intents.values())} examples from "
              f"{len(unmapped_intents)} unmapped intents")

    return prepared


def add_domain_examples(prepared: dict, domain_examples: list[dict]) -> dict:
    """Add hand-written domain examples to the training split."""
    domain_formatted = []
    for ex in domain_examples:
        domain_formatted.append({
            "text": ex["text"],
            "original_intent": ex["intent"],
            "department": ex["department"],
            "lang": ex["lang"],
            "source": "domain_examples",
        })

    prepared["train"].extend(domain_formatted)
    print(f"  Added {len(domain_formatted)} domain examples to training set")
    return prepared


def print_statistics(prepared: dict) -> None:
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total = 0
    for split_name, examples in prepared.items():
        print(f"\n--- {split_name.upper()} ---")
        print(f"  Total examples: {len(examples)}")
        total += len(examples)

        # Department distribution
        dept_counts = Counter(ex["department"] for ex in examples)
        print("  Department distribution:")
        for dept in config.DEPARTMENTS:
            count = dept_counts.get(dept, 0)
            bar = "█" * (count // 5) if count > 0 else ""
            print(f"    {dept:25s}: {count:5d}  {bar}")

        # Source distribution
        source_counts = Counter(ex["source"] for ex in examples)
        print("  Source distribution:")
        for source, count in source_counts.most_common():
            print(f"    {source:25s}: {count:5d}")

        # Language distribution
        lang_counts = Counter(ex["lang"] for ex in examples)
        print("  Language distribution:")
        for lang, count in lang_counts.most_common():
            print(f"    {lang:25s}: {count:5d}")

    print(f"\n  TOTAL across all splits: {total}")
    print("=" * 60)


def save_dataset(prepared: dict, output_path: str) -> None:
    """Save the prepared dataset to JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(prepared, f, ensure_ascii=False, indent=2)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nDataset saved to {output_path} ({file_size:.2f} MB)")


def main():
    """Main entry point for data preparation."""
    print("TechnoHome Customer Support - Data Preparation")
    print("=" * 60)

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Step 1: Load domain examples
    print("\n[1/4] Loading domain examples...")
    domain_examples_path = os.path.join(config.DATA_DIR, "domain_examples.json")
    domain_examples = load_domain_examples(domain_examples_path)

    # Step 2: Download CLINC150
    print("\n[2/4] Downloading CLINC150...")
    dataset = download_clinc150()

    # Step 3: Map intents to departments
    print("\n[3/4] Mapping intents to departments...")
    prepared = map_to_departments(dataset)

    # Step 4: Add domain examples
    print("\n[4/4] Adding domain examples to training set...")
    prepared = add_domain_examples(prepared, domain_examples)

    # Print statistics
    print_statistics(prepared)

    # Save prepared dataset
    output_path = os.path.join(config.DATA_DIR, "prepared_dataset.json")
    save_dataset(prepared, output_path)

    print("\nData preparation complete!")


if __name__ == "__main__":
    main()
