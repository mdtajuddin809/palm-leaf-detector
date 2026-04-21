import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import datetime
import base64
import os

# --- Page Config ---
st.set_page_config(page_title="Palm Leaf Detector", page_icon="🌿", layout="wide")

# --- Load background image as base64 ---
def get_base64_image(image_path):
    """Load a local image and convert to base64 for CSS embedding."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# Try to load the custom background image
# Place "background.png" (your farm image) in the same folder as app.py
bg_base64 = get_base64_image("background.png")

if bg_base64:
    bg_css = f"url('data:image/png;base64,{bg_base64}')"
else:
    # Fallback to Unsplash if image not found
    bg_css = "url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6')"

# --- Custom CSS ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&family=Lato:wght@300;400;700&display=swap');

    .stApp {{
        background-image: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
                          {bg_css};
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    html, body, [class*="css"] {{
        font-family: 'Lato', sans-serif;
    }}

    h1, h2, h3 {{
        font-family: 'Merriweather', serif;
    }}

    .main-title {{
        text-align: center;
        color: #a8f0a0;
        font-size: 2.5rem;
        font-family: 'Merriweather', serif;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
        padding: 0.5rem 0;
        letter-spacing: 1px;
    }}

    .subtitle {{
        text-align: center;
        color: #d4f7d4;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        opacity: 0.85;
    }}

    .card {{
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(168, 240, 160, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(6px);
    }}

    .treatment-card {{
        background: rgba(20, 80, 20, 0.55);
        border-left: 5px solid #5dba5d;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        color: #e8ffe8;
    }}

    .severity-low    {{ border-left-color: #5dba5d; }}
    .severity-medium {{ border-left-color: #f0c040; }}
    .severity-high   {{ border-left-color: #e05050; }}

    .badge {{
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        margin-left: 8px;
    }}
    .badge-green  {{ background: #2e7d32; color: #fff; }}
    .badge-yellow {{ background: #f9a825; color: #000; }}
    .badge-red    {{ background: #c62828; color: #fff; }}

    .stMetric {{ background: rgba(255,255,255,0.08); border-radius: 10px; padding: 0.5rem; }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ─────────────────────────────────────────────
# TREATMENT DICTIONARY  (expanded)
# Each "en"/"te" is a LIST → rendered as numbered steps
# ─────────────────────────────────────────────
treatments = {
    # ── Original deficiencies ──
    "Boron": {
        "en": [
            "Apply Boron fertilizer (Borax) at 1–2 kg/tree/year to the soil in recommended quantity.",
            "Spray 0.1–0.2% Borax solution on leaves for faster recovery.",
            "Improve soil organic matter with compost and maintain proper moisture.",
            "Avoid waterlogging — excess moisture blocks Boron uptake.",
        ],
        "te": [
            "బోరాన్ ఎరువు (బోరాక్స్)ను సరైన మోతాదులో (1–2 కి.గ్రా/చెట్టు/సంవత్సరం) నేలలో వేయండి.",
            "ఆకులపై 0.1–0.2% బోరాక్స్ ద్రావణం స్ప్రే చేయండి.",
            "నేలలో సేంద్రియ పదార్థం పెంచి తగిన తేమ ఉంచండి.",
            "నీరు నిల్వ కాకుండా జాగ్రత్తగా ఉండండి.",
        ],
        "severity": "medium",
        "icon": "🪨",
        "causes": "Sandy soils, high pH (>7), or excessive rainfall leaching.",
        "symptoms": "Yellowing of young leaves, bent/curved new fronds, poor fruit set.",
    },
    "Nitrogen": {
        "en": [
            "Apply Nitrogen fertilizers like Urea or Ammonium Sulfate at 1–2 kg/tree/year.",
            "Split into 3–4 doses per year for better absorption.",
            "Improve irrigation to help nutrient absorption — Nitrogen leaches quickly in sandy soils.",
            "Add organic compost to enrich soil fertility.",
            "For quick results, foliar spray of 2% Urea solution on leaves.",
        ],
        "te": [
            "యూరియా లేదా అమోనియం సల్ఫేట్ వంటి నైట్రోజన్ ఎరువులు వేయండి (1–2 కి.గ్రా/చెట్టు/సంవత్సరం).",
            "సంవత్సరానికి 3–4 విడతలుగా ఇవ్వండి.",
            "పోషకాలు గ్రహించడానికి నీటి సరఫరా మెరుగుపరచండి.",
            "నేల సారాన్ని పెంచడానికి సేంద్రియ ఎరువులు వేయండి.",
            "వేగంగా ఫలితం కోసం 2% యూరియా ద్రావణం ఆకులపై స్ప్రే చేయండి.",
        ],
        "severity": "high",
        "icon": "🌱",
        "causes": "Poor soil, heavy rain leaching, sandy or acidic soils.",
        "symptoms": "Pale yellow-green older leaves, stunted growth, small fruit bunches.",
    },
    "Magnesium": {
        "en": (
            "Apply Magnesium Sulfate (Epsom Salt) at 1–2 kg/tree/year to soil. "
            "For quick relief, spray 2% Magnesium Sulfate solution on leaves. "
            "Avoid excessive Potassium application as it competes with Mg uptake. "
            "Use dolomite lime to correct soil Mg levels on acidic soils."
        ),
        "te": (
            "మెగ్నీషియం సల్ఫేట్ (ఎప్సమ్ సాల్ట్) 1–2 కి.గ్రా/చెట్టు/సంవత్సరం నేలకు ఇవ్వండి. "
            "వేగంగా ఫలితం కోసం 2% ద్రావణం ఆకులపై స్ప్రే చేయండి. అధిక పొటాషియం ఇవ్వకండి. "
            "ఆమ్ల నేలలకు డోలమైట్ సున్నం ఉపయోగించండి."
        ),
        "severity": "medium",
        "icon": "🧂",
        "causes": "Leaching in sandy soils, high Potassium/Calcium competition.",
        "symptoms": "Yellowing between leaf veins (interveinal chlorosis) on older fronds.",
    },
    "Kalium": {
        "en": (
            "Apply Muriate of Potash (MOP) or Sulfate of Potash (SOP) at 1.5–3 kg/tree/year. "
            "Split application into 2–3 doses per year. Ensure balanced NPK ratio. "
            "Potassium is critical for drought tolerance and fruit quality in oil palms. "
            "Avoid applying in waterlogged conditions."
        ),
        "te": (
            "మ్యూరియేట్ ఆఫ్ పొటాష్ (MOP) లేదా సల్ఫేట్ ఆఫ్ పొటాష్ (SOP) 1.5–3 కి.గ్రా/చెట్టు/"
            "సంవత్సరం ఇవ్వండి. 2–3 విడతలుగా ఇవ్వడం మంచిది. సరైన NPK నిష్పత్తి పాటించండి. "
            "నీరు నిల్వ ఉన్నప్పుడు ఎరువులు వేయకండి."
        ),
        "severity": "medium",
        "icon": "⚗️",
        "causes": "Leaching, low soil organic matter, excessive rainfall.",
        "symptoms": "Orange/necrotic spotting on older leaves, reduced bunch weight.",
    },
    "Phosphorus": {
        "en": (
            "Apply Single Super Phosphate (SSP) or Triple Super Phosphate (TSP) at 1–2 kg/tree/year. "
            "Incorporate into the soil around the feeding root zone. Phosphorus availability drops "
            "in very acidic or alkaline soils — correct soil pH first (target 5.5–6.5). "
            "Mycorrhizal inoculants can significantly improve phosphorus uptake."
        ),
        "te": (
            "సింగిల్ సూపర్ ఫాస్ఫేట్ (SSP) లేదా ట్రిపుల్ సూపర్ ఫాస్ఫేట్ (TSP) 1–2 కి.గ్రా/చెట్టు/"
            "సంవత్సరం ఇవ్వండి. చెట్టు వేర్ల చుట్టూ నేలలో కలపండి. నేల pH సరిచేయండి (5.5–6.5 లక్ష్యం). "
            "మైకోరైజల్ ఇనాక్యులెంట్లు ఫాస్ఫరస్ గ్రహణాన్ని మెరుగుపరుస్తాయి."
        ),
        "severity": "high",
        "icon": "🔥",
        "causes": "Acidic/alkaline soils, waterlogged conditions, low organic matter.",
        "symptoms": "Dark green or purplish tinge on older leaves, weak root development.",
    },
    "Iron": {
        "en": (
            "Apply Ferrous Sulfate (FeSO₄) as foliar spray (0.5–1%) or chelated iron (Fe-EDTA) "
            "to soil at 50–100 g/tree. Iron deficiency is often caused by high soil pH rather than "
            "actual iron shortage — correct pH to 5.5–6.5. Avoid over-liming. "
            "Repeat spray every 2–3 weeks until new growth appears green."
        ),
        "te": (
            "ఫెర్రస్ సల్ఫేట్ (0.5–1%) ఆకులపై స్ప్రే చేయండి లేదా చెలేటెడ్ ఐరన్ (Fe-EDTA) "
            "50–100 గ్రా/చెట్టు నేలకు ఇవ్వండి. నేల pH ఎక్కువగా ఉంటే ఇనుము అందుబాటులో ఉండదు — "
            "pH 5.5–6.5కి సరిచేయండి. కొత్త పెరుగుదల ఆకుపచ్చగా వచ్చేంత వరకు ప్రతి 2–3 వారాలకు స్ప్రే చేయండి."
        ),
        "severity": "medium",
        "icon": "🔩",
        "causes": "High soil pH (>7), waterlogging, excess phosphorus or manganese.",
        "symptoms": "Young leaves turn yellow-white while veins stay green (interveinal chlorosis).",
    },
    "Manganese": {
        "en": (
            "Apply Manganese Sulfate (25–50 g/tree) to soil or as foliar spray (0.5%). "
            "Manganese deficiency is common in high-pH or waterlogged soils. "
            "Correct soil pH to 5.5–6.5 for better availability. "
            "Avoid excessive iron or zinc applications which compete with Mn uptake."
        ),
        "te": (
            "మాంగనీస్ సల్ఫేట్ (25–50 గ్రా/చెట్టు) నేలకు ఇవ్వండి లేదా 0.5% ద్రావణం ఆకులపై స్ప్రే చేయండి. "
            "అధిక pH లేదా నీరు నిల్వ ఉన్న నేలలో మాంగనీస్ కొరత సాధారణం. "
            "నేల pH 5.5–6.5కి సరిచేయండి. అధిక ఇనుము లేదా జింక్ ఇవ్వకండి."
        ),
        "severity": "low",
        "icon": "🌀",
        "causes": "High pH soils, waterlogging, excessive liming.",
        "symptoms": "Interveinal chlorosis on young leaves; grayish-green or pale patches.",
    },
    "Zinc": {
        "en": (
            "Apply Zinc Sulfate (ZnSO₄) at 25–50 g/tree to soil or spray 0.5% solution on leaves. "
            "Zinc is often deficient in calcareous (limestone) or heavily leached soils. "
            "Mix Zinc Sulfate with lime before foliar application to reduce leaf burn. "
            "Zinc is important for enzyme activity, growth hormone, and seed development."
        ),
        "te": (
            "జింక్ సల్ఫేట్ 25–50 గ్రా/చెట్టు నేలకు ఇవ్వండి లేదా 0.5% ద్రావణం ఆకులపై స్ప్రే చేయండి. "
            "సున్నపురాయి నేలలు లేదా అధికంగా కడిగిన నేలలలో జింక్ కొరత సాధారణం. "
            "ఆకు కాలుపు నివారించేందుకు స్ప్రే కు ముందు సున్నం కలపండి. "
            "జింక్ ఎంజైమ్ కార్యకలాపాలు మరియు పెరుగుదల హార్మోన్లకు అవసరం."
        ),
        "severity": "low",
        "icon": "⚡",
        "causes": "High pH, calcareous soils, excessive phosphorus.",
        "symptoms": "Small, narrow leaves; shortened internodes; mottled yellowish young leaves.",
    },
    "Copper": {
        "en": (
            "Apply Copper Sulfate (CuSO₄) at 10–20 g/tree or use Bordeaux mixture. "
            "Copper deficiency is rare but occurs in heavily leached or peaty soils. "
            "Use copper-based fungicides as dual-purpose treatment. "
            "Avoid excessive copper — it is toxic at high concentrations and kills beneficial soil microbes."
        ),
        "te": (
            "కాపర్ సల్ఫేట్ 10–20 గ్రా/చెట్టు ఇవ్వండి లేదా బోర్డో మిశ్రమం ఉపయోగించండి. "
            "రాగి కొరత అరుదు కానీ అధికంగా కడిగిన లేదా పీట్ నేలలలో వస్తుంది. "
            "రాగి ఆధారిత శిలీంద్రనాశకాలు రెండు పనులు చేస్తాయి. "
            "అధిక రాగి విషపూరితమవుతుంది — జాగ్రత్తగా వాడండి."
        ),
        "severity": "low",
        "icon": "🟤",
        "causes": "Peaty or organic soils, heavily leached sandy soils.",
        "symptoms": "Bluish-green wilting of young fronds, necrotic leaf tips.",
    },
    "Calcium": {
        "en": (
            "Apply agricultural lime (CaCO₃) or gypsum (CaSO₄) at 1–3 kg/tree. "
            "Calcium improves soil structure and root development. "
            "Ensure adequate soil moisture — Calcium moves with water in the plant. "
            "Avoid excessive ammonium-based fertilizers which compete with Ca uptake."
        ),
        "te": (
            "వ్యవసాయ సున్నం (CaCO₃) లేదా జిప్సమ్ (CaSO₄) 1–3 కి.గ్రా/చెట్టు ఇవ్వండి. "
            "కాల్షియం నేల నిర్మాణాన్ని మరియు వేర్ల అభివృద్ధిని మెరుగుపరుస్తుంది. "
            "తగినంత నీటి నిర్వహణ పాటించండి — మొక్కలో కాల్షియం నీటితో కదులుతుంది. "
            "అమోనియం ఆధారిత ఎరువులు అధికంగా ఇవ్వకండి."
        ),
        "severity": "medium",
        "icon": "🦴",
        "causes": "Acidic soils, low CEC soils, drought, excessive potassium.",
        "symptoms": "Tip burn on young leaves, poor root growth, bud necrosis.",
    },
    "Sulfur": {
        "en": (
            "Apply Ammonium Sulfate or Gypsum to add sulfur to soil. "
            "Sulfur deficiency looks similar to Nitrogen but affects younger leaves first. "
            "Elemental sulfur can be used but takes time to convert in soil. "
            "Sulfur is critical for protein synthesis and chlorophyll formation."
        ),
        "te": (
            "అమోనియం సల్ఫేట్ లేదా జిప్సమ్ ద్వారా నేలకు సల్ఫర్ ఇవ్వండి. "
            "సల్ఫర్ కొరత నైట్రోజన్ లా కనిపిస్తుంది కానీ ముందు చిన్న ఆకులపై వస్తుంది. "
            "మూల సల్ఫర్ నేలలో మారడానికి సమయం పడుతుంది. "
            "సల్ఫర్ ప్రోటీన్ సంశ్లేషణ మరియు క్లోరోఫిల్ నిర్మాణానికి అవసరం."
        ),
        "severity": "medium",
        "icon": "🌑",
        "causes": "Low organic matter, high rainfall leaching, sandy soils.",
        "symptoms": "Uniform yellowing of young leaves, similar to N deficiency.",
    },
    "Ganoderma": {
        "en": (
            "No chemical cure exists — management is key. Remove and destroy infected palm immediately. "
            "Apply Trichoderma-based biocontrol agents around the root zone. "
            "Avoid wounding the trunk during harvesting. Sterilize tools between palms. "
            "Apply Hexaconazole or Propiconazole as preventive trunk injection on neighboring trees. "
            "Improve drainage as waterlogging favors disease spread."
        ),
        "te": (
            "రసాయన చికిత్స లేదు — నిర్వహణ చాలా ముఖ్యం. సోకిన చెట్టును వెంటనే తొలగించి నాశనం చేయండి. "
            "వేర్ల చుట్టూ ట్రైకోడెర్మా జీవ నియంత్రణ ఔషధాలు వేయండి. "
            "కోత సమయంలో కాండంకు గాయాలు చేయకండి. సాధనాలు క్రిమిరహితం చేయండి. "
            "పక్కన ఉన్న చెట్లకు హెక్సాకోనజోల్ ట్రంక్ ఇంజెక్షన్ ఇవ్వండి. "
            "నీటి నిల్వ నిరోధించేందుకు మురుగు నీటి నిర్వహణ మెరుగుపరచండి."
        ),
        "severity": "high",
        "icon": "🍄",
        "causes": "Ganoderma boninense fungus; spreads through soil contact and root wounds.",
        "symptoms": "Yellowing/browning of fronds from base upward, stem rot, fungal brackets at base.",
    },
    "Bud_Rot": {
        "en": (
            "Remove and destroy infected spear leaves. Apply Metalaxyl or Fosetyl-aluminium fungicide "
            "directly into the bud. Improve drainage to reduce waterlogging. "
            "Avoid mechanical damage to the spear leaf. Apply copper-based fungicide as preventive spray. "
            "Monitor neighboring palms closely."
        ),
        "te": (
            "సోకిన స్పియర్ ఆకులను తొలగించి నాశనం చేయండి. మెటాలాక్సిల్ లేదా ఫోసెటిల్-అల్యుమినియం "
            "శిలీంద్రనాశకాన్ని నేరుగా మొగ్గలో వేయండి. మురుగు నీటి నిర్వహణ మెరుగుపరచండి. "
            "స్పియర్ ఆకుకు యాంత్రిక నష్టం చేయకండి. నివారణగా రాగి ఆధారిత శిలీంద్రనాశకం స్ప్రే చేయండి."
        ),
        "severity": "high",
        "icon": "💀",
        "causes": "Phytophthora palmivora; worsened by waterlogging and physical damage.",
        "symptoms": "Rotting of central spear, foul odor, spear pulls out easily.",
    },
    "Leaf_Spot": {
        "en": (
            "Apply Mancozeb or Copper Oxychloride fungicide (2 g/L water) as foliar spray. "
            "Remove severely infected leaves and destroy them. "
            "Improve air circulation by proper spacing and pruning. "
            "Avoid overhead irrigation — wet foliage promotes fungal spread. "
            "Repeat spray every 2–3 weeks during wet season."
        ),
        "te": (
            "మాంకోజెబ్ లేదా కాపర్ ఆక్సీక్లోరైడ్ శిలీంద్రనాశకం (2 గ్రా/లీ నీరు) ఆకులపై స్ప్రే చేయండి. "
            "తీవ్రంగా సోకిన ఆకులను తొలగించి నాశనం చేయండి. "
            "సరైన అంతర సాగు మరియు కత్తిరింపుతో గాలి ప్రసరణ మెరుగుపరచండి. "
            "పైనుండి నీటి పారుదల నివారించండి. వర్షాకాలంలో ప్రతి 2–3 వారాలకు స్ప్రే చేయండి."
        ),
        "severity": "medium",
        "icon": "🟡",
        "causes": "Fungal pathogens (Cercospora, Pestalotiopsis); high humidity.",
        "symptoms": "Brown/yellow spots with yellow halo on older leaves.",
    },
    "Healthy": {
        "en": (
            "Leaf is healthy. No treatment required. "
            "Continue regular fertilization schedule (NPK + micronutrients). "
            "Maintain proper irrigation and drainage. Monitor monthly for early signs of deficiency or disease."
        ),
        "te": (
            "ఆకు ఆరోగ్యంగా ఉంది. చికిత్స అవసరం లేదు. "
            "నిర్ణీత NPK + సూక్ష్మపోషక ఎరువుల కార్యక్రమాన్ని కొనసాగించండి. "
            "సరైన నీటి మరియు మురుగు నీటి నిర్వహణ పాటించండి. "
            "తొందరగా గుర్తించేందుకు నెలవారీ పర్యవేక్షణ చేయండి."
        ),
        "severity": "low",
        "icon": "✅",
        "causes": "—",
        "symptoms": "No visible symptoms.",
    },
}

SEVERITY_BADGE = {
    "low":    ('<span class="badge badge-green">Low</span>', "severity-low"),
    "medium": ('<span class="badge badge-yellow">Medium</span>', "severity-medium"),
    "high":   ('<span class="badge badge-red">High Risk</span>', "severity-high"),
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 Palm Leaf Detector")
    st.markdown("---")

    language = st.selectbox("🌐 Language / భాష", ["English", "Telugu"])
    lang_key = "en" if language == "English" else "te"

    conf_threshold = st.slider("🎯 Confidence Threshold", 0.10, 1.0, 0.25, 0.05)

    show_causes    = st.toggle("Show Causes",   value=True)
    show_symptoms  = st.toggle("Show Symptoms", value=True)

    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

    st.markdown("---")
    st.markdown(
        f"<small style='color:#aaa;'>🕒 {datetime.datetime.now().strftime('%d %b %Y, %H:%M')}</small>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    "<p class='main-title'>🌿 Palm Leaf Nutrient & Disease Detection</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='subtitle'>AI-powered diagnosis with treatment recommendations in English & Telugu</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if uploaded_file is not None:
    file_bytes   = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image        = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("🔍 Analyzing leaf..."):
        results = model(image, conf=conf_threshold)

    result_img = results[0].plot()
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # Images
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(original_rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Detection Output")
        st.image(result_rgb, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Download button
    result_pil = Image.fromarray(result_rgb)
    with st.sidebar:
        st.markdown("---")
        st.download_button(
            label="⬇️ Download Result",
            data=cv2.imencode('.jpg', cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name=f"detection_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
        )

    # ── Detection Details ──
    st.markdown("---")
    st.subheader("📊 Detection Details")

    boxes = results[0].boxes

    if len(boxes) == 0:
        st.warning("⚠️ No detections found. Try lowering the Confidence Threshold in the sidebar.")
    else:
        # Metrics row
        num_cols = min(len(boxes), 4)
        cols = st.columns(num_cols)
        shown_classes = {}   # label → highest conf

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]

            with cols[i % num_cols]:
                st.metric(f"{label}", f"{conf*100:.1f}%")

            if label not in shown_classes or conf > shown_classes[label]:
                shown_classes[label] = conf

        # Treatment cards
        st.markdown("### 🌱 Treatment Recommendations")

        for label, conf in shown_classes.items():
            info = treatments.get(label)
            if info is None:
                for key in treatments:
                    if key.lower() == label.lower():
                        info = treatments[key]
                        break

            if info is None:
                st.info(f"ℹ️ **{label}** — No treatment info available.")
                continue

            severity          = info.get("severity", "low")
            badge_html, sev_class = SEVERITY_BADGE[severity]
            icon              = info.get("icon", "🌿")
            treatment_steps   = info[lang_key]
            causes            = info.get("causes", "—")
            symptoms          = info.get("symptoms", "—")

            if isinstance(treatment_steps, list):
                steps_html = "".join(
                    f"<li style='margin-bottom:5px;'>{step}</li>"
                    for step in treatment_steps
                )
            else:
                steps_html = f"<li>{treatment_steps}</li>"

            treat_label = "Treatment / చికిత్స" if language == "Telugu" else "Treatment"

            st.markdown(
                f"""
                <div class="treatment-card {sev_class}">
                    <strong style="font-size:1.1rem;">{icon} {label}</strong>
                    {badge_html}
                    <span style="color:#aef2ae; font-size:0.85rem; float:right;">Confidence: {conf*100:.1f}%</span>
                    <br><br>
                    <b>{treat_label}:</b>
                    <ol style="margin-top:8px; padding-left:1.3rem; line-height:1.8;">
                        {steps_html}
                    </ol>
                """,
                unsafe_allow_html=True,
            )

            if show_causes:
                st.markdown(
                    f"<div class='treatment-card' style='margin-top:4px;background:rgba(0,40,0,0.4);'>"
                    f"<b>🔎 Causes:</b> {causes}</div>",
                    unsafe_allow_html=True,
                )
            if show_symptoms:
                st.markdown(
                    f"<div class='treatment-card' style='margin-top:4px;background:rgba(0,40,0,0.4);'>"
                    f"<b>🩺 Symptoms:</b> {symptoms}</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("")

        # Summary table
        st.markdown("### 📋 Summary Table")
        summary_data = {
            "Deficiency / Disease": [],
            "Severity": [],
            "Confidence (%)": [],
        }
        for label, conf in shown_classes.items():
            info = treatments.get(label, {})
            summary_data["Deficiency / Disease"].append(
                f"{info.get('icon','🌿')} {label}"
            )
            summary_data["Severity"].append(info.get("severity", "—").capitalize())
            summary_data["Confidence (%)"].append(f"{conf*100:.1f}")

        import pandas as pd
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

else:
    st.markdown(
        """
        <div style='text-align:center; margin-top: 4rem; color: #c8f7c8;'>
            <h2>👆 Upload an image to begin</h2>
            <p style='opacity:0.7;'>Supports JPG, PNG, JPEG · AI model analyzes nutrient deficiencies & diseases</p>
            <p style='opacity:0.5; font-size:0.85rem;'>Detects: Boron · Nitrogen · Magnesium · Potassium · Phosphorus · Iron ·
            Manganese · Zinc · Copper · Calcium · Sulfur · Ganoderma · Bud Rot · Leaf Spot · Healthy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )