"""
Product name spell correction module for TechnoHome Customer Support Ticket Router.
Corrects misspelled product names using Levenshtein distance and a product lexicon.
"""

import os
import sys
import json
import re

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.
    Pure Python implementation — no external dependencies required.
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class ProductNameCorrector:
    """
    Corrects misspelled product names in customer support text.

    Loads a product lexicon from data/product_lexicon.json and uses
    Levenshtein distance to find the closest match for potentially
    misspelled product names.
    """

    # Maximum edit distance to consider a match
    MAX_DISTANCE_THRESHOLD = 3

    # Minimum word length to attempt correction (skip short words)
    MIN_WORD_LENGTH = 4

    def __init__(self, lexicon_path: str = None):
        """
        Initialize the corrector with a product lexicon.

        Args:
            lexicon_path: Path to product_lexicon.json.
                          Defaults to data/product_lexicon.json from project root.
        """
        if lexicon_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            lexicon_path = os.path.join(project_root, "data", "product_lexicon.json")

        self.products = []
        self.known_words = {}         # lowercase word -> correct form
        self.misspelling_map = {}     # misspelling -> correct form

        self._load_lexicon(lexicon_path)

    def _load_lexicon(self, path: str):
        """Load the product lexicon and build lookup tables."""
        if not os.path.exists(path):
            print(f"[ProductNameCorrector] Warning: Lexicon not found at {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both direct list and {"products": [...]} format
        if isinstance(data, dict) and "products" in data:
            products = data["products"]
        elif isinstance(data, list):
            products = data
        else:
            products = []

        self.products = products

        for product in products:
            # Register product name
            name = product.get("name", "")
            if name:
                self.known_words[name.lower()] = name

            # Register aliases
            aliases = product.get("aliases", [])
            if isinstance(aliases, dict):
                # Flatten dict of alias lists (e.g., {"tr": [...], "en": [...]})
                alias_list = []
                for lang_aliases in aliases.values():
                    if isinstance(lang_aliases, list):
                        alias_list.extend(lang_aliases)
                    else:
                        alias_list.append(lang_aliases)
                aliases = alias_list

            for alias in aliases:
                if alias:
                    self.known_words[alias.lower()] = alias

            # Register common misspellings
            misspellings = product.get("common_misspellings", {})
            if isinstance(misspellings, dict):
                for correct_form, variants in misspellings.items():
                    # Support both formats:
                    # {correct: [misspelling1, misspelling2]} and {misspelling: correct}
                    if isinstance(variants, list):
                        for misspelled in variants:
                            self.misspelling_map[misspelled.lower()] = correct_form
                    elif isinstance(variants, str):
                        self.misspelling_map[correct_form.lower()] = variants
            elif isinstance(misspellings, list):
                for misspelled in misspellings:
                    self.misspelling_map[misspelled.lower()] = name

            # Register category
            category = product.get("category", "")
            if category:
                self.known_words[category.lower()] = category

    def _find_best_match(self, word: str) -> str:
        """
        Find the best matching known word using Levenshtein distance.

        Args:
            word: The potentially misspelled word (already lowercased).

        Returns:
            The corrected word, or the original word if no close match found.
        """
        if len(word) < self.MIN_WORD_LENGTH:
            return word

        # Check exact match first
        if word in self.known_words:
            return word

        # Check misspelling map for direct lookup
        if word in self.misspelling_map:
            return self.misspelling_map[word]

        # Find closest match via Levenshtein distance
        best_match = None
        best_distance = float("inf")

        all_candidates = list(self.known_words.keys()) + list(self.misspelling_map.keys())

        for candidate in all_candidates:
            # Skip if length difference is already > threshold (quick reject)
            if abs(len(candidate) - len(word)) > self.MAX_DISTANCE_THRESHOLD:
                continue

            dist = _levenshtein_distance(word, candidate)
            if dist < best_distance and dist <= self.MAX_DISTANCE_THRESHOLD:
                best_distance = dist
                if candidate in self.misspelling_map:
                    best_match = self.misspelling_map[candidate]
                else:
                    best_match = self.known_words.get(candidate, candidate)

        if best_match is not None:
            if isinstance(best_match, str):
                return best_match.lower()
            elif isinstance(best_match, list):
                return best_match[0].lower() if best_match else word

        return word

    def correct(self, text: str) -> str:
        """
        Correct misspelled product names in the given text.

        Args:
            text: Input text that may contain misspelled product names.

        Returns:
            Text with corrected product names.
        """
        if not text or not isinstance(text, str):
            return text

        if not self.known_words and not self.misspelling_map:
            # No lexicon loaded, return as-is
            return text

        # Tokenize while preserving structure (spaces, punctuation)
        # Split on word boundaries but keep separators
        tokens = re.split(r"(\s+|[.,;:!?()\"'\-/])", text)

        corrected_tokens = []
        for token in tokens:
            # Only attempt correction on actual words (not whitespace/punctuation)
            stripped = token.strip()
            if stripped and re.match(r"^[a-zA-ZçğıöşüÇĞİÖŞÜ]+$", stripped):
                corrected = self._find_best_match(stripped.lower())
                # Preserve original casing style if the word wasn't corrected
                if corrected == stripped.lower():
                    corrected_tokens.append(token)
                else:
                    # Apply the corrected form
                    if token[0].isupper() and len(corrected) > 0:
                        corrected = corrected[0].upper() + corrected[1:]
                    corrected_tokens.append(corrected)
            else:
                corrected_tokens.append(token)

        return "".join(corrected_tokens)


if __name__ == "__main__":
    corrector = ProductNameCorrector()

    test_cases = [
        "bulaşık makinesu bozuldu",
        "çamşır makinesi çalışmıyor",
        "elektirikli süpürge sesi çok yüksek",
        "my dishwasher is broken",
        "mikrodalgam ısınmıyor",
    ]

    for text in test_cases:
        corrected = corrector.correct(text)
        print(f"Input:  {text}")
        print(f"Output: {corrected}")
        print()
