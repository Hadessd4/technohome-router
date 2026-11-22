"""
TechnoHome Customer Support Ticket Router - Configuration
SWE015 Homework - Use Case 2
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "saved_models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create dirs if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ─── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "clinc/clinc_oos"
DATASET_CONFIG = "plus"  # Includes extra OOS examples
RANDOM_SEED = 42

# ─── Department Mapping ──────────────────────────────────────────────────────
# Maps CLINC150 intents to our corporate departments
DEPARTMENT_MAP = {
    # 🔧 Technical Support
    "technical_support": {
        "display_name": "🔧 Technical Support",
        "clinc_intents": [
            "car_rental", "oil_change_how", "oil_change_when",
            "tire_change", "tire_pressure",
        ],
        "description": "Hardware issues, troubleshooting, error codes",
    },
    # 📦 Order & Delivery
    "order_delivery": {
        "display_name": "📦 Order & Delivery",
        "clinc_intents": [
            "order_status", "order", "cancel",
            "shopping_list", "shopping_list_update",
        ],
        "description": "Order tracking, shipping, delivery scheduling",
    },
    # 💳 Billing & Payments
    "billing_payments": {
        "display_name": "💳 Billing & Payments",
        "clinc_intents": [
            "bill_balance", "bill_due", "pay_bill",
            "credit_limit", "credit_limit_change",
            "credit_score", "report_fraud",
        ],
        "description": "Invoices, payment issues, refunds",
    },
    # 🔄 Returns & Warranty
    "returns_warranty": {
        "display_name": "🔄 Returns & Warranty",
        "clinc_intents": [
            "return", "exchange_rate", "insurance",
            "insurance_change",
        ],
        "description": "Returns, warranty claims, product exchanges",
    },
    # 📋 Product Information
    "product_info": {
        "display_name": "📋 Product Information",
        "clinc_intents": [
            "ingredient_substitution", "ingredients_list",
            "calories", "nutrition_info",
            "recipe", "food_last",
        ],
        "description": "Specs, compatibility, product recommendations",
    },
    # 🛠 Installation & Setup
    "installation_setup": {
        "display_name": "🛠 Installation & Setup",
        "clinc_intents": [
            "plug_type", "smart_home",
            "how_busy", "directions",
        ],
        "description": "Installation guides, connection help, setup",
    },
    # 📞 Account Management
    "account_management": {
        "display_name": "📞 Account Management",
        "clinc_intents": [
            "account_blocked", "pin_change", "reset_settings",
            "change_user_name", "change_language",
            "change_accent", "change_speed",
        ],
        "description": "Account updates, password resets, preferences",
    },
    # 🤖 Human Agent (Fallback)
    "human_agent": {
        "display_name": "🤖 Human Agent",
        "clinc_intents": ["oos"],  # Out-of-scope
        "description": "Unrecognized or low-confidence queries",
    },
}

# Reverse mapping: CLINC intent → department
INTENT_TO_DEPARTMENT = {}
for dept, info in DEPARTMENT_MAP.items():
    for intent in info["clinc_intents"]:
        INTENT_TO_DEPARTMENT[intent] = dept

# All department labels
DEPARTMENTS = list(DEPARTMENT_MAP.keys())
NUM_DEPARTMENTS = len(DEPARTMENTS)
DEPT_TO_IDX = {dept: i for i, dept in enumerate(DEPARTMENTS)}
IDX_TO_DEPT = {i: dept for dept, i in DEPT_TO_IDX.items()}

# ─── Model Hyperparameters ───────────────────────────────────────────────────
# TF-IDF + LogReg
TFIDF_MAX_FEATURES = 10000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000

# FastText + FFN
FASTTEXT_DIM = 300
FFN_HIDDEN_DIM = 256
FFN_DROPOUT = 0.3
FFN_LEARNING_RATE = 0.01
FFN_MOMENTUM = 0.9
FFN_EPOCHS = 30
FFN_BATCH_SIZE = 32

# BERT
BERT_MODEL_NAME = "dbmdz/bert-base-turkish-cased"  # BERTurk
BERT_MAX_LENGTH = 64
BERT_LEARNING_RATE = 2e-5
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 32
BERT_WARMUP_STEPS = 100

# Distillation
STUDENT_MODEL_NAME = "distilbert-base-multilingual-cased"
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA = 0.5
DISTILL_EPOCHS = 3
DISTILL_LEARNING_RATE = 5e-5

# ─── Confidence Threshold ────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.5  # Below this → route to human agent

# ─── Latency Measurement ─────────────────────────────────────────────────────
LATENCY_RUNS = 100  # Number of runs for averaging latency
