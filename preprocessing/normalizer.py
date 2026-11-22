"""
Text normalization module for TechnoHome Customer Support Ticket Router.
Handles Turkish and English text preprocessing while preserving product model codes.
"""

import os
import sys
import re
import unicodedata

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TextNormalizer:
    """
    Normalizes customer support text input.

    - Extracts product model codes (e.g., DW-6534, RF-9200) before lowercasing
    - Lowercases text
    - Removes excessive punctuation and repeated characters
    - Strips URLs, emails, and extra whitespace
    - Preserves Turkish characters (ş, ç, ğ, ı, ö, ü, İ, Ş, Ç, Ğ, Ö, Ü)
    """

    # Turkish characters that must be preserved during normalization
    TURKISH_CHARS = set("şçğıöüİŞÇĞÖÜ")

    # Regex for product model codes like DW-6534, RF-9200, WM-4100, AC-7700
    MODEL_CODE_PATTERN = re.compile(r"[A-Z]{2,3}-\d{3,4}")

    # URL pattern
    URL_PATTERN = re.compile(
        r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE
    )

    # Email pattern
    EMAIL_PATTERN = re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE
    )

    # Excessive punctuation: 3+ of the same punctuation character in a row
    EXCESSIVE_PUNCTUATION_PATTERN = re.compile(r"([!?.,;:…\-])\1{2,}")

    # Multiple whitespace
    MULTI_WHITESPACE_PATTERN = re.compile(r"\s+")

    # Repeated characters (3+ of the same letter) — handles Turkish chars too
    REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")

    def __init__(self):
        pass

    def _extract_model_codes(self, text: str) -> list:
        """Extract product model codes from text before any transformation."""
        codes = self.MODEL_CODE_PATTERN.findall(text)
        # Deduplicate while preserving order
        seen = set()
        unique_codes = []
        for code in codes:
            if code not in seen:
                seen.add(code)
                unique_codes.append(code)
        return unique_codes

    def _remove_urls(self, text: str) -> str:
        """Strip URLs from text."""
        return self.URL_PATTERN.sub(" ", text)

    def _remove_emails(self, text: str) -> str:
        """Strip email addresses from text."""
        return self.EMAIL_PATTERN.sub(" ", text)

    def _turkish_lower(self, text: str) -> str:
        """
        Lowercase text with proper Turkish character handling.
        Turkish 'İ' -> 'i', 'I' -> 'ı' (but we only handle İ->i here;
        standard I->ı requires locale awareness, so we keep it simple).
        """
        # Handle Turkish-specific uppercase-to-lowercase mappings
        result = text.replace("İ", "i").replace("I", "ı")
        # Then standard lowercase for the rest
        result = result.lower()
        return result

    def _reduce_repeated_chars(self, text: str) -> str:
        """
        Reduce 3+ repeated characters to 1 (for letters) or 2 (for punctuation).
        E.g., 'yardımmm' -> 'yardım', 'harika!!!' -> 'harika!!'
        """
        def _replace_repeated(match):
            char = match.group(1)
            # For punctuation, keep 2; for letters, keep 1
            if char in "!?.,;:-…":
                return char * 2
            return char

        return self.REPEATED_CHAR_PATTERN.sub(_replace_repeated, text)

    def _clean_punctuation(self, text: str) -> str:
        """Remove excessive punctuation (3+ same punctuation -> 2)."""
        return self.EXCESSIVE_PUNCTUATION_PATTERN.sub(r"\1\1", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple whitespace characters into a single space."""
        return self.MULTI_WHITESPACE_PATTERN.sub(" ", text).strip()

    def normalize(self, text: str) -> dict:
        """
        Normalize the input text.

        Args:
            text: Raw customer support text input.

        Returns:
            dict with keys:
                - normalized_text (str): The cleaned, lowercased text.
                - model_codes (list): Extracted product model codes (uppercase).
        """
        if not text or not isinstance(text, str):
            return {"normalized_text": "", "model_codes": []}

        # Step 1: Extract model codes BEFORE any text transformation
        model_codes = self._extract_model_codes(text)

        # Step 2: Remove URLs and emails
        result = self._remove_urls(text)
        result = self._remove_emails(result)

        # Step 3: Lowercase with Turkish-aware handling
        result = self._turkish_lower(result)

        # Step 4: Reduce repeated characters (e.g., "yardımmm" -> "yardım")
        result = self._reduce_repeated_chars(result)

        # Step 5: Clean excessive punctuation
        result = self._clean_punctuation(result)

        # Step 6: Normalize whitespace
        result = self._normalize_whitespace(result)

        return {
            "normalized_text": result,
            "model_codes": model_codes,
        }


if __name__ == "__main__":
    # Quick test
    normalizer = TextNormalizer()

    test_cases = [
        "DW-6534 modelim bozuldu LÜTFEN YARDIMMM!!!",
        "Merhaba, RF-9200 ve AC-7700 ürünlerini karşılaştırabilir misiniz?",
        "info@technohome.com adresine mail attım https://technohome.com/destek",
        "ÇAMAŞIR MAKİNESİ WM-4100 çalışmıyorrrrr",
        "Hello I need help with my OV-3300 oven pleeease",
    ]

    for text in test_cases:
        result = normalizer.normalize(text)
        print(f"Input:  {text}")
        print(f"Output: {result}")
        print()
