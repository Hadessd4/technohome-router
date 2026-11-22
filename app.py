"""
TechnoHome Customer Support Ticket Router - Streamlit Application
SWE015 Homework - Use Case 2
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import config

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TechnoHome Support Router",
    layout="wide",
    page_icon="🏢",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Corporate dark-blue theme */
    .stApp {
        background-color: #0a1628;
    }
    .main-header {
        background: linear-gradient(135deg, #0d1f3c 0%, #1a3a6b 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #1e4d8c;
    }
    .main-header h1 {
        color: #ffffff;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: #8cb4e0;
        margin: 0.25rem 0 0 0;
        font-size: 1rem;
    }
    .model-card {
        background: #101d33;
        border: 1px solid #1e4d8c;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .model-card h4 {
        color: #5ba3f5;
        margin-top: 0;
    }
    .routing-box {
        background: linear-gradient(135deg, #0d2b1a 0%, #143d25 100%);
        border: 2px solid #2ecc71;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .routing-box h3 {
        color: #2ecc71;
    }
    .routing-box-warning {
        background: linear-gradient(135deg, #2b1d0d 0%, #3d2914 100%);
        border: 2px solid #f39c12;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1rem;
    }
    .routing-box-warning h3 {
        color: #f39c12;
    }
    .status-loaded {
        color: #2ecc71;
        font-weight: bold;
    }
    .status-not-loaded {
        color: #e74c3c;
        font-weight: bold;
    }
    .preprocess-box {
        background: #101d33;
        border: 1px solid #1e4d8c;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Department Icons ─────────────────────────────────────────────────────────
DEPT_ICONS = {
    "technical_support": "🔧",
    "order_delivery": "📦",
    "billing_payments": "💳",
    "returns_warranty": "🔄",
    "product_info": "📋",
    "installation_setup": "🛠",
    "account_management": "📞",
    "human_agent": "🤖",
}

# ─── Sample Messages ─────────────────────────────────────────────────────────
SAMPLE_MESSAGES = {
    "-- Select a sample message --": "",
    "🇹🇷 Bulaşık makinesi arızası": "Bulaşık makinem su boşaltmıyor, model DW-6534",
    "🇹🇷 Sipariş takibi": "Siparişim 3 gündür gelmedi, kargo nerede?",
    "🇹🇷 Fatura sorunu": "Bu ayki faturamda fazla ücret var, iade istiyorum",
    "🇹🇷 Garanti süreci": "Çamaşır makinem garanti kapsamında mı, WM-4100 model",
    "🇹🇷 Ürün bilgisi": "RF-9200 buzdolabının enerji sınıfı nedir?",
    "🇹🇷 Kurulum yardımı": "Klima montajı nasıl yapılır, AC-7700",
    "🇹🇷 Hesap yönetimi": "Şifremi unuttum, hesabıma giremiyorum",
    "🇬🇧 Technical issue": "My dishwasher DW-6534 is leaking water from the bottom",
    "🇬🇧 Order tracking": "Where is my order? It's been 5 days since I placed it",
    "🇬🇧 Billing question": "I was charged twice for my last purchase",
    "🇬🇧 Warranty claim": "I want to return my vacuum cleaner VC-8800, it stopped working",
    "🇬🇧 Product specs": "What is the capacity of the OV-3300 oven?",
    "🇹🇷 Selam!": "Merhaba, nasılsınız?",
    "🇬🇧 Hello!": "Hello, how are you?",
}


# ─── Model Loading (Cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_tfidf_model():
    """Load TF-IDF + Logistic Regression model."""
    try:
        from models.baseline_tfidf import TFIDFClassifier
        model = TFIDFClassifier()
        model.load()
        return model
    except Exception as e:
        st.warning(f"⚠️ TF-IDF model could not be loaded: {e}")
        return None


@st.cache_resource
def load_ffn_model():
    """Load FastText + Feedforward NN model."""
    try:
        from models.feedforward_nn import FastTextFFNClassifier
        model = FastTextFFNClassifier()
        model.load()
        return model
    except Exception as e:
        st.warning(f"⚠️ FFN model could not be loaded: {e}")
        return None


@st.cache_resource
def load_bert_model():
    """Load BERT model."""
    try:
        from models.bert_classifier import BERTClassifier
        model = BERTClassifier()
        model.load()
        return model
    except Exception as e:
        st.warning(f"⚠️ BERT model could not be loaded: {e}")
        return None


@st.cache_resource
def load_quantized_model():
    """Load Distilled + Quantized model."""
    try:
        from models.distill_and_quantize import QuantizedClassifier
        model = QuantizedClassifier()
        model.load()
        return model
    except Exception as e:
        st.warning(f"⚠️ Quantized model could not be loaded: {e}")
        return None


@st.cache_resource
def load_preprocessors():
    """Load preprocessing components."""
    components = {}
    try:
        from preprocessing.normalizer import TextNormalizer
        components["normalizer"] = TextNormalizer()
    except Exception as e:
        st.warning(f"⚠️ Normalizer could not be loaded: {e}")
        components["normalizer"] = None

    try:
        from preprocessing.spell_corrector import ProductNameCorrector
        components["spell_corrector"] = ProductNameCorrector()
    except Exception as e:
        st.warning(f"⚠️ Spell corrector could not be loaded: {e}")
        components["spell_corrector"] = None

    try:
        from preprocessing.small_talk import SmallTalkDetector
        components["small_talk"] = SmallTalkDetector()
    except Exception as e:
        st.warning(f"⚠️ Small talk detector could not be loaded: {e}")
        components["small_talk"] = None

    return components


def load_product_lexicon():
    """Load product lexicon for lookup."""
    lexicon_path = config.DATA_DIR / "product_lexicon.json"
    if lexicon_path.exists():
        with open(lexicon_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def predict_with_timing(model, text):
    """Run prediction and measure latency."""
    start = time.perf_counter()
    result = model.predict(text)
    latency_ms = (time.perf_counter() - start) * 1000
    return result, latency_ms


def render_model_prediction(col, title, model, text, threshold):
    """Render a single model's prediction in a column."""
    with col:
        st.markdown(f"#### {title}")
        if model is None:
            st.warning("Model not loaded")
            return None

        try:
            result, latency = predict_with_timing(model, text)
            # Handle both dict and tuple return formats
            if isinstance(result, dict):
                department = result.get("department", "unknown")
                confidence = result.get("confidence", 0.0)
            elif isinstance(result, (tuple, list)):
                department, confidence = result[0], result[1]
            else:
                department = str(result)
                confidence = 0.0
            icon = DEPT_ICONS.get(department, "❓")

            st.metric(
                label="Department",
                value=f"{icon} {department}",
            )
            st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.1%}")
            st.caption(f"⏱️ Latency: {latency:.1f} ms")

            if confidence < threshold:
                st.warning("→ 🤖 Routed to Human Agent")
                return {"department": "human_agent", "confidence": confidence, "latency": latency}
            else:
                st.success(f"→ {icon} {config.DEPARTMENT_MAP.get(department, {}).get('display_name', department)}")
                return {"department": department, "confidence": confidence, "latency": latency}
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏢 TechnoHome – Intelligent Support Router</h1>
    <p>AI-Powered Customer Support Ticket Classification & Routing System</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
preprocessors = load_preprocessors()
tfidf_model = load_tfidf_model()
ffn_model = load_ffn_model()
bert_model = load_bert_model()
quantized_model = load_quantized_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/smart-home-2.png", width=80)
    st.markdown("### 🏢 TechnoHome")
    st.markdown(
        "Welcome to the **TechnoHome Intelligent Support Router**. "
        "This system uses 4 different ML models to classify and route "
        "customer support tickets to the correct department."
    )

    st.divider()

    # Confidence threshold
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Below this confidence, tickets are routed to a human agent.",
    )

    st.divider()

    # Model status
    st.markdown("### 📡 Model Status")
    models_status = {
        "TF-IDF + LogReg": tfidf_model is not None,
        "FastText + FFN": ffn_model is not None,
        "BERTurk": bert_model is not None,
        "Quantized": quantized_model is not None,
    }
    for name, loaded in models_status.items():
        if loaded:
            st.markdown(f'<span class="status-loaded">✅ {name}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-not-loaded">❌ {name}</span>', unsafe_allow_html=True)

    st.divider()

    # Sample messages
    st.markdown("### 💬 Sample Messages")
    selected_sample = st.selectbox(
        "Choose a sample message:",
        options=list(SAMPLE_MESSAGES.keys()),
        index=0,
    )

# ─── Main Area ────────────────────────────────────────────────────────────────
# Determine initial text from sample selection
initial_text = SAMPLE_MESSAGES.get(selected_sample, "")

user_input = st.text_area(
    "📝 Enter your support message:",
    value=initial_text,
    height=100,
    placeholder="e.g., bulaşık makinem su boşaltmıyor, model DW-6534",
)

route_clicked = st.button("🚀 Route Ticket", type="primary", use_container_width=True)

if route_clicked and user_input.strip():
    st.divider()

    # ── Step 1: Preprocessing ─────────────────────────────────────────────
    st.markdown("### 🔍 Preprocessing Results")

    normalized_text = user_input
    model_codes = []
    corrected_text = user_input

    col_pre1, col_pre2 = st.columns(2)

    with col_pre1:
        if preprocessors.get("normalizer"):
            norm_result = preprocessors["normalizer"].normalize(user_input)
            normalized_text = norm_result.get("normalized_text", user_input)
            model_codes = norm_result.get("model_codes", [])
            st.markdown("**Normalized Text:**")
            st.code(normalized_text, language=None)
            if model_codes:
                st.markdown(f"**Extracted Model Codes:** `{'`, `'.join(model_codes)}`")
        else:
            st.warning("Normalizer not available")

    with col_pre2:
        if preprocessors.get("spell_corrector"):
            corrected_text = preprocessors["spell_corrector"].correct(normalized_text)
            if corrected_text != normalized_text:
                st.markdown("**Spell Corrections:**")
                st.code(f"{normalized_text}\n→ {corrected_text}", language=None)
            else:
                st.markdown("**Spell Check:** No corrections needed ✅")
        else:
            st.warning("Spell corrector not available")

    st.divider()

    # ── Step 2: Small Talk Check ──────────────────────────────────────────
    small_talk_reply = None
    if preprocessors.get("small_talk"):
        small_talk_reply = preprocessors["small_talk"].check(corrected_text)

    if small_talk_reply:
        st.markdown("### 💬 Small Talk Detected")
        st.info(f"🤖 **TechnoHome Assistant:** {small_talk_reply}")
    else:
        # ── Step 3: Model Predictions ─────────────────────────────────────
        st.markdown("### 🧪 Model Predictions")

        process_text = corrected_text

        col1, col2, col3, col4 = st.columns(4)

        results = {}
        results["tfidf"] = render_model_prediction(
            col1, "📊 TF-IDF + LogReg", tfidf_model, process_text, threshold
        )
        results["ffn"] = render_model_prediction(
            col2, "🧠 FastText + FFN", ffn_model, process_text, threshold
        )
        results["bert"] = render_model_prediction(
            col3, "🤖 BERT", bert_model, process_text, threshold
        )
        results["quantized"] = render_model_prediction(
            col4, "⚡ Quantized", quantized_model, process_text, threshold
        )

        st.divider()

        # ── Step 4: Final Routing Decision (based on quantized model) ─────
        st.markdown("### 🎯 Final Routing Decision")

        final_result = results.get("quantized")

        if final_result is None:
            # Fallback to BERT, then FFN, then TF-IDF
            for fallback_key in ["bert", "ffn", "tfidf"]:
                if results.get(fallback_key) is not None:
                    final_result = results[fallback_key]
                    st.caption(f"⚠️ Using fallback model ({fallback_key}) for routing decision")
                    break

        if final_result:
            dept = final_result["department"]
            conf = final_result["confidence"]
            dept_display = config.DEPARTMENT_MAP.get(dept, {}).get("display_name", dept)
            dept_desc = config.DEPARTMENT_MAP.get(dept, {}).get("description", "")
            icon = DEPT_ICONS.get(dept, "❓")

            if dept == "human_agent" or conf < threshold:
                st.markdown(f"""
                <div class="routing-box-warning">
                    <h3>🤖 Routed to Human Agent</h3>
                    <p style="color: #f5a623; font-size: 1.1rem;">
                        Confidence ({conf:.1%}) is below threshold ({threshold:.0%}).
                        This ticket will be reviewed by a human support agent.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="routing-box">
                    <h3>{icon} {dept_display}</h3>
                    <p style="color: #8fd4a8; font-size: 1.1rem;">
                        Confidence: {conf:.1%} — {dept_desc}
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ No models available for routing. Please train models first.")

        # ── Step 5: Product Info Lookup ────────────────────────────────────
        if model_codes:
            st.divider()
            st.markdown("### 📦 Product Information")
            lexicon = load_product_lexicon()
            if lexicon:
                products = lexicon.get("products", lexicon) if isinstance(lexicon, dict) else lexicon
                # Handle both list and dict formats
                if isinstance(products, list):
                    product_map = {p.get("code", ""): p for p in products}
                elif isinstance(products, dict):
                    product_map = {}
                    for key, val in products.items():
                        if isinstance(val, dict) and "code" in val:
                            product_map[val["code"]] = val
                        elif isinstance(val, dict):
                            product_map[key] = val
                else:
                    product_map = {}

                for code in model_codes:
                    product = product_map.get(code)
                    if product:
                        with st.expander(f"📋 {code} — {product.get('name', 'Unknown Product')}", expanded=True):
                            pcol1, pcol2 = st.columns(2)
                            with pcol1:
                                st.markdown(f"**Name:** {product.get('name', 'N/A')}")
                                st.markdown(f"**Category:** {product.get('category', 'N/A')}")
                                st.markdown(f"**Code:** `{code}`")
                            with pcol2:
                                aliases = product.get("aliases", [])
                                if aliases:
                                    st.markdown(f"**Aliases:** {', '.join(aliases) if isinstance(aliases, list) else aliases}")
                    else:
                        st.info(f"ℹ️ Product code `{code}` not found in lexicon.")
            else:
                st.info("ℹ️ Product lexicon not available.")

elif route_clicked:
    st.warning("⚠️ Please enter a support message to route.")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "🏢 TechnoHome Intelligent Support Router | "
    "SWE015 Homework – Use Case 2 | "
    "Powered by TF-IDF, FastText, BERTurk & Quantized Models"
)
