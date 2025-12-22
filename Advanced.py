import streamlit as st
from PIL import Image
import io
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess

# ReportLab for PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🥬 Egyptian Vegetable Classifier",
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60
HIGH_CONF_THRESHOLD = 0.70
UNKNOWN_LABEL = "Unknown (Non-Vegetable or Unsupported Input)"

CLASS_NAMES = [
    "Bean", "Bitter Gourd", "Bottle Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

# ---------------- NUTRITION & RECIPES ----------------
NUTRITION_INFO = {
    "Bean": {"Calories": "347 kcal", "Protein": "21g", "Carbs": "63g", "Fiber": "16g", "Fat": "1.2g",
             "Vitamin C": "4.8mg (5%)", "Iron": "8.2mg (46%)", "Potassium": "1480mg (31%)",
             "Note": "Values for dry beans. Excellent plant-based protein source."},
    "Bitter Gourd": {"Calories": "17 kcal", "Protein": "1g", "Carbs": "4g", "Fiber": "2.6g", "Fat": "0.2g",
                     "Vitamin C": "84mg (93%)", "Vitamin A": "470 IU", "Folate": "72µg",
                     "Note": "Extremely high in Vitamin C."},
    "Bottle Gourd": {"Calories": "14 kcal", "Protein": "0.6g", "Carbs": "3.4g", "Fiber": "0.5g", "Fat": "0.1g",
                      "Vitamin C": "10mg", "Potassium": "150mg", "Water": "~96%",
                      "Note": "Very low calorie and highly hydrating."},
    "Brinjal": {"Calories": "25 kcal", "Protein": "1g", "Carbs": "6g", "Fiber": "3g", "Fat": "0.2g",
                "Vitamin C": "2.2mg", "Potassium": "229mg", "Anthocyanins": "High",
                "Note": "Eggplant skin is rich in antioxidants."},
    "Broccoli": {"Calories": "34 kcal", "Protein": "2.8g", "Carbs": "7g", "Fiber": "2.6g", "Fat": "0.4g",
                 "Vitamin C": "89mg (99%)", "Vitamin K": "102µg (85%)", "Folate": "63µg",
                 "Note": "One of the most nutrient-dense vegetables!"},
    "Cabbage": {"Calories": "25 kcal", "Protein": "1.3g", "Carbs": "6g", "Fiber": "2.5g", "Fat": "0.1g",
                "Vitamin C": "36mg (40%)", "Vitamin K": "76µg (63%)",
                "Note": "Great for gut health and immunity."},
    "Capsicum": {"Calories": "31 kcal", "Protein": "1g", "Carbs": "6g", "Fiber": "2.1g", "Fat": "0.3g",
                 "Vitamin C": "128mg (142%)", "Vitamin A": "3131 IU (63%)",
                 "Note": "Red bell peppers are vitamin powerhouses."},
    "Carrot": {"Calories": "41 kcal", "Protein": "0.9g", "Carbs": "10g", "Fiber": "2.8g", "Fat": "0.2g",
               "Vitamin A": "16706 IU (334%)", "Potassium": "320mg",
               "Note": "Outstanding source of beta-carotene."},
    "Cauliflower": {"Calories": "25 kcal", "Protein": "1.9g", "Carbs": "5g", "Fiber": "2g", "Fat": "0.3g",
                    "Vitamin C": "48mg (53%)", "Folate": "57µg",
                    "Note": "Popular low-carb rice/potato substitute."},
    "Cucumber": {"Calories": "15 kcal", "Protein": "0.7g", "Carbs": "4g", "Fiber": "0.5g", "Fat": "0.1g",
                 "Vitamin K": "16µg", "Water": "~95%",
                 "Note": "Extremely refreshing and hydrating."},
    "Papaya": {"Calories": "43 kcal", "Protein": "0.5g", "Carbs": "11g", "Fiber": "1.7g", "Fat": "0.3g",
               "Vitamin C": "61mg (68%)", "Vitamin A": "950 IU",
               "Note": "Contains papain, a digestive enzyme."},
    "Potato": {"Calories": "77 kcal", "Protein": "2g", "Carbs": "18g", "Fiber": "2.2g", "Fat": "0.1g",
               "Vitamin C": "20mg (22%)", "Potassium": "425mg (9%)",
               "Note": "With skin – excellent energy source."},
    "Pumpkin": {"Calories": "26 kcal", "Protein": "1g", "Carbs": "6.5g", "Fiber": "0.5g", "Fat": "0.1g",
                "Vitamin A": "8513 IU (170%)", "Vitamin C": "9mg",
                "Note": "Rich in immune-boosting beta-carotene."},
    "Radish": {"Calories": "16 kcal", "Protein": "0.7g", "Carbs": "3.4g", "Fiber": "1.6g", "Fat": "0.1g",
               "Vitamin C": "15mg (16%)", "Potassium": "233mg",
               "Note": "Crisp, peppery, and low in calories."},
    "Tomato": {"Calories": "18 kcal", "Protein": "0.9g", "Carbs": "3.9g", "Fiber": "1.2g", "Fat": "0.2g",
               "Vitamin C": "14mg (15%)", "Lycopene": "High",
               "Note": "Powerful antioxidant for heart health."}
}

RECIPE_SUGGESTIONS = {
    "Bean": ["Ful Medames", "Ta'meya", "Besarah"],
    "Bitter Gourd": ["Mahshi Karela", "Bitter Gourd Salad", "Pickled Bitter Gourd"],
    "Bottle Gourd": ["Mahshi Koosa", "Bottle Gourd Soup", "Torly"],
    "Brinjal": ["Mahshi Betengan", "Baba Ghanoush", "Mesa'a'ah"],
    "Broccoli": ["Broccoli bel Zabadi", "Broccoli Mahshi", "Broccoli with Tahini"],
    "Cabbage": ["Mahshi Malfouf", "Cabbage Salad", "Torly Cabbage"],
    "Capsicum": ["Mahshi Filfil", "Filfil bel Tamatem", "Salata Filfil"],
    "Carrot": ["Salatet Jazar", "Carrot Torly", "Carrot Pickles"],
    "Cauliflower": ["Arnabeet Mekhalel", "Fried Arnabeet", "Cauliflower Torly"],
    "Cucumber": ["Khiar bel Zabadi", "Salata Baladi", "Pickled Cucumbers"],
    "Papaya": ["Papaya wa Asal", "Papaya Juice", "Papaya Salad"],
    "Potato": ["Batates Mahmerah", "Potato Torly", "Batates bel Furn"],
    "Pumpkin": ["Qara'a bel Laban", "Pumpkin Mahshi", "Sweet Pumpkin Puree"],
    "Radish": ["Feggous bel Torshi", "Radish Salad", "Radish with Ful"],
    "Tomato": ["Salata Baladi", "Shakshuka", "Tomato Salsa"]
}

# ---------------- CSS ----------------
st.markdown("""
<style>
h1 {font-size:42px;color:#2E8B57;font-family:Arial;}
h2 {color:#228B22;margin-top:20px;margin-bottom:10px;}
h3 {color:#006400;}
.card {background-color:#ffffff;border-radius:12px;padding:20px;margin-bottom:20px;box-shadow:0 6px 18px rgba(0,0,0,0.07);}
.stButton>button {background-color:#2E8B57;color:white;font-weight:bold;border-radius:8px;padding:0.6em 1em;}
.stButton>button:hover {background-color:#1e593c;}
.st-expander {background-color:#f7fff7;border-left:4px solid #2E8B57;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    try:
        cnn = load_model("cnn_model (1).h5", compile=False)
    except Exception:
        st.error("Custom CNN model not found.")
        cnn = None
    try:
        effnet = load_model("efficientnetb0_finetuned.keras", compile=False)
    except Exception:
        st.error("EfficientNet model not found.")
        effnet = None
    return cnn, effnet

cnn_model, effnet_model = load_models()

# ---------------- PDF FUNCTION ----------------
def generate_pdf_report(image, image_size, top3_details, agreement_text, inference_times):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Egyptian Vegetable Classification Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Uploaded Image</b>", styles["Heading2"]))
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    rl_img = RLImage(img_buffer, width=10*cm, height=10*cm)
    elements.append(rl_img)
    elements.append(Paragraph(f"Resolution: {image_size[0]} × {image_size[1]} pixels", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Model Agreement</b>", styles["Heading2"]))
    elements.append(Paragraph(agreement_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Top Predictions</b>", styles["Heading2"]))
    for model_name, preds in top3_details.items():
        elements.append(Paragraph(f"<b>{model_name}</b> (Time: {inference_times.get(model_name, 0):.3f}s)", styles["Heading3"]))
        table_data = [["Rank", "Vegetable", "Confidence (%)"]]
        for i, (cls, conf) in enumerate(preds, 1):
            table_data.append([i, cls, f"{conf*100:.2f}"])
        table = Table(table_data, colWidths=[2*cm, 7*cm, 4*cm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (2,1), (-1,-1), 'CENTER')
        ]))
        elements.append(table)
        elements.append(Spacer(1, 8))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph("Generated using Egyptian cuisine recipes.", styles["Italic"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- MAIN APP ----------------
st.title("🥬 Egyptian Vegetable Classifier")
st.markdown("*Upload a vegetable → Get authentic Egyptian recipes & nutrition!*")

available_models = []
if cnn_model: available_models.append("Custom CNN")
if effnet_model: available_models.append("EfficientNetB0")
if len(available_models) == 2: available_models.append("Compare Both")

model_choice = st.selectbox("Choose model", available_models,
                            index=len(available_models)-1 if "Compare Both" in available_models else 0)

# ---------------- IMAGE INPUT ----------------
uploaded_file = st.file_uploader("Upload a vegetable photo", type=["jpg","png","jpeg","webp"])
camera_file = st.camera_input("Or take a photo using your camera")

# Select input image
image_input = None
if camera_file is not None:
    image_input = Image.open(camera_file).convert("RGB")
elif uploaded_file is not None:
    image_input = Image.open(uploaded_file).convert("RGB")

# ---------------- HELPER FUNCTIONS ----------------
def get_top_predictions(preds, top_k=3):
    indices = np.argsort(preds)[-top_k:][::-1]
    return [(CLASS_NAMES[i], float(preds[i])) for i in indices]

def plot_top_predictions(model_name, preds):
    classes = [c for c,_ in preds]
    confs = [c*100 for _,c in preds]
    fig,ax=plt.subplots(figsize=(5,2))
    bars=ax.barh(classes[::-1],confs[::-1],color='#228B22')
    ax.set_xlim(0,100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"{model_name}")
    for bar in bars:
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                f"{bar.get_width():.1f}%", va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ---------------- PREDICTION LOGIC ----------------
if image_input:
    resized = image_input.resize(IMG_SIZE)
    cnn_batch = np.expand_dims(np.array(resized),0)
    eff_batch = efficient_preprocess(np.expand_dims(np.array(resized),0).astype(np.float32))

    top3_details = {}
    inference_times = {}
    best_predictions = {}

    if model_choice in ["Custom CNN","Compare Both"] and cnn_model:
        start=time.time()
        preds=cnn_model.predict(cnn_batch,verbose=0)[0]
        inference_times["Custom CNN"]=time.time()-start
        top3=get_top_predictions(preds)
        top3_details["Custom CNN"]=top3
        best_predictions["Custom CNN"]=top3[0][0]

    if model_choice in ["EfficientNetB0","Compare Both"] and effnet_model:
        start=time.time()
        preds=effnet_model.predict(eff_batch,verbose=0)[0]
        inference_times["EfficientNetB0"]=time.time()-start
        top3=get_top_predictions(preds)
        top3_details["EfficientNetB0"]=top3
        best_predictions["EfficientNetB0"]=top3[0][0]

    # Determine final class
    if len(best_predictions)==2:
        if best_predictions["Custom CNN"]==best_predictions["EfficientNetB0"]:
            final_class=best_predictions["Custom CNN"]
            final_conf=max(d[0][1] for d in top3_details.values())
            agreement="🟢 Both models agree!"
        else:
            final_class=None
            final_conf=0
            agreement="🟡 Models disagree"
    else:
        final_class=list(best_predictions.values())[0] if best_predictions else None
        final_conf=list(top3_details.values())[0][0][1] if top3_details else 0
        agreement="Single model result"

    # Check unknown
    is_unknown = (final_class is None or final_conf < CONFIDENCE_THRESHOLD or agreement=="🟡 Models disagree")
    if is_unknown:
        final_class = UNKNOWN_LABEL

    # ---------------- UI ----------------
    col1,col2,col3 = st.columns([2,2.5,3],gap="large")

    with col1:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.image(image_input,caption="Uploaded Image",use_column_width=True)
        st.caption(f"Resolution: {image_input.size[0]} × {image_input.size[1]} px")
        st.markdown('</div>',unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("🍽 Nutrition & Recipes")
        if final_class==UNKNOWN_LABEL:
            st.error("❌ Unsupported input detected")
            st.info("📌 Please upload a clear image of a vegetable from the supported list.")
        else:
            info=NUTRITION_INFO.get(final_class,{})
            recipes=RECIPE_SUGGESTIONS.get(final_class,[])
            st.metric("Calories (per 100g)",info.get("Calories","N/A"))
            with st.expander("Key Nutrients"):
                for nut,val in info.items():
                    if nut!="Note":
                        st.markdown(f"• **{nut}**: {val}")
            with st.expander("🍴 Egyptian Recipe Suggestions"):
                for recipe in recipes:
                    st.markdown(f"• {recipe}")
        st.markdown('</div>',unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.subheader("📊 Model Predictions")
        for model_name,preds in top3_details.items():
            best_c,best_cf=preds[0]
            st.markdown(f"**{model_name}** ({agreement})")
            st.progress(best_cf)
            st.caption(f"{best_cf*100:.1f}% confidence • {inference_times[model_name]:.3f}s")
            plot_top_predictions(model_name,preds)
        st.markdown('</div>',unsafe_allow_html=True)
        if final_class==UNKNOWN_LABEL:
            st.warning("⚠️ Prediction rejected due to low confidence or model disagreement.")

    # PDF Download
    if top3_details:
        pdf_buffer=generate_pdf_report(image_input,image_input.size,top3_details,agreement,inference_times)
        st.download_button(
            "📄 Download Egyptian Recipe Report (PDF)",
            data=pdf_buffer,
            file_name=f"egyptian_vegetable_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Upload a vegetable image or take a photo to discover authentic Egyptian recipes!")
    st.markdown("**Supported vegetables:** " + ", ".join(CLASS_NAMES))
