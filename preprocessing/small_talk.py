"""
Small talk detection module for TechnoHome Customer Support Ticket Router.
Detects greetings, thanks, goodbyes, and casual conversation in Turkish and English,
and returns corporate-branded responses mentioning TechnoHome.
"""

import os
import sys
import re
from typing import Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SmallTalkDetector:
    """
    ELIZA-style small talk detector for customer support.

    Detects greeting, thanks, goodbye, and how-are-you patterns in both
    Turkish and English, and responds with corporate-branded TechnoHome messages.
    """

    # Each category has a list of (compiled_regex, response) tuples.
    # Patterns are checked in order; first match wins.

    PATTERNS = {
        "greeting_tr": [
            (
                re.compile(r"\b(merhaba|selam|selamlar|merhabalar|hey|heyy|sa|selamün\s*aleyküm)\b", re.IGNORECASE),
                [
                    "Merhaba! TechnoHome müşteri hizmetlerine hoş geldiniz. Size nasıl yardımcı olabilirim?",
                    "Merhabalar! TechnoHome'a hoş geldiniz. Bugün size nasıl destek olabilirim?",
                ],
            ),
            (
                re.compile(r"\b(günaydın|hayırlı\s*sabahlar)\b", re.IGNORECASE),
                [
                    "Günaydın! TechnoHome müşteri hizmetlerine hoş geldiniz. Size nasıl yardımcı olabilirim?",
                ],
            ),
            (
                re.compile(r"\b(iyi\s*akşamlar)\b", re.IGNORECASE),
                [
                    "İyi akşamlar! TechnoHome müşteri hizmetlerine hoş geldiniz. Size nasıl yardımcı olabilirim?",
                ],
            ),
            (
                re.compile(r"\b(iyi\s*günler)\b", re.IGNORECASE),
                [
                    "İyi günler! TechnoHome müşteri hizmetleri olarak size nasıl yardımcı olabilirim?",
                ],
            ),
        ],
        "greeting_en": [
            (
                re.compile(r"\b(hello|hi|hey|howdy|good\s*morning|good\s*afternoon|good\s*evening|greetings)\b", re.IGNORECASE),
                [
                    "Hello! Welcome to TechnoHome customer support. How can I help you today?",
                    "Hi there! Welcome to TechnoHome. How may I assist you?",
                ],
            ),
        ],
        "thanks_tr": [
            (
                re.compile(r"\b(teşekkür\s*ederim|teşekkürler|sağ\s*ol|sağ\s*olun|çok\s*teşekkürler|teşekkür\s*ederiz)\b", re.IGNORECASE),
                [
                    "Rica ederim! TechnoHome olarak her zaman yardımcı olmaktan mutluluk duyarız. Başka bir sorunuz var mı?",
                    "Ne demek, her zaman! TechnoHome müşteri hizmetleri olarak size yardımcı olmak bizim görevimiz.",
                ],
            ),
        ],
        "thanks_en": [
            (
                re.compile(r"\b(thanks?|thank\s*you|thanks?\s*a\s*lot|much\s*appreciated|ty|thx)\b", re.IGNORECASE),
                [
                    "You're welcome! At TechnoHome, we're always happy to help. Is there anything else I can assist you with?",
                    "My pleasure! TechnoHome customer support is here for you anytime.",
                ],
            ),
        ],
        "goodbye_tr": [
            (
                re.compile(r"\b(hoşça\s*kal|görüşürüz|iyi\s*günler\s*dilerim|bay\s*bay|güle\s*güle|hoşça\s*kalın|kendinize\s*iyi\s*bakın)\b", re.IGNORECASE),
                [
                    "Hoşça kalın! TechnoHome'u tercih ettiğiniz için teşekkür ederiz. İyi günler dileriz!",
                    "Görüşmek üzere! TechnoHome müşteri hizmetleri her zaman yanınızda. İyi günler!",
                ],
            ),
        ],
        "goodbye_en": [
            (
                re.compile(r"\b(bye|goodbye|good\s*bye|see\s*you|take\s*care|farewell|have\s*a\s*nice\s*day|cya)\b", re.IGNORECASE),
                [
                    "Goodbye! Thank you for choosing TechnoHome. Have a wonderful day!",
                    "Take care! TechnoHome customer support is always here when you need us.",
                ],
            ),
        ],
        "how_are_you_tr": [
            (
                re.compile(r"\b(nasılsın|nasılsınız|naber|ne\s*haber|naber\s*nasılsın|keyifler\s*nasıl|iyi\s*misin|iyi\s*misiniz|n['']?aber)\b", re.IGNORECASE),
                [
                    "İyiyim, teşekkür ederim! TechnoHome müşteri hizmetleri olarak size yardımcı olmaya hazırım. Nasıl yardımcı olabilirim?",
                    "Çok iyiyim, sorduğunuz için teşekkürler! TechnoHome'da size en iyi hizmeti sunmak için buradayım.",
                ],
            ),
        ],
        "how_are_you_en": [
            (
                re.compile(r"\b(how\s*are\s*you|how('s|\s*is)\s*it\s*going|how\s*do\s*you\s*do|what'?s\s*up|how('s|\s*is)\s*everything)\b", re.IGNORECASE),
                [
                    "I'm doing great, thank you for asking! I'm here at TechnoHome to help you. What can I do for you today?",
                    "All good here at TechnoHome! How may I assist you?",
                ],
            ),
        ],
    }

    def __init__(self):
        # Flatten all pattern categories into an ordered list for matching
        self._pattern_list = []
        for category, patterns in self.PATTERNS.items():
            for regex, responses in patterns:
                self._pattern_list.append((category, regex, responses))

    def _is_small_talk_only(self, text: str, match: re.Match) -> bool:
        """
        Check if the text is primarily small talk (not a substantive question
        with a greeting attached). A short text that is mostly the matched
        pattern is considered pure small talk.
        """
        # If text is short (roughly just the greeting/etc.), treat as small talk
        remaining = text[:match.start()].strip() + " " + text[match.end():].strip()
        remaining = remaining.strip()

        # If there's substantial remaining text (more than a few words), it's
        # probably a real question with a greeting prefix — still return the
        # small talk response but the router can decide to process further.
        # For this module, we detect and respond regardless.
        return True

    def check(self, text: str) -> Optional[str]:
        """
        Check if the given text is small talk.

        Args:
            text: The input text to check.

        Returns:
            A corporate-branded response string if small talk is detected,
            None otherwise.
        """
        if not text or not isinstance(text, str):
            return None

        cleaned = text.strip()
        if not cleaned:
            return None

        for category, regex, responses in self._pattern_list:
            match = regex.search(cleaned)
            if match:
                # Use a deterministic selection based on text length
                # to keep behavior reproducible
                idx = len(cleaned) % len(responses)
                return responses[idx]

        return None


if __name__ == "__main__":
    detector = SmallTalkDetector()

    test_cases = [
        "Merhaba",
        "selam nasılsın",
        "Hello there",
        "teşekkür ederim çok yardımcı oldunuz",
        "thank you so much",
        "hoşça kalın",
        "goodbye",
        "how are you doing today?",
        "nasılsınız?",
        "DW-6534 modelim bozuldu yardım edin",  # Not small talk
        "günaydın",
        "bye bye",
        "naber",
    ]

    for text in test_cases:
        result = detector.check(text)
        if result:
            print(f"Input:    {text}")
            print(f"Response: {result}")
        else:
            print(f"Input:    {text}")
            print(f"Response: [Not small talk]")
        print()
