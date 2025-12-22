import streamlit as st
from PIL import Image
import io
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as efficient_preprocess
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D


# ReportLab for PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm

if "gradcam_image" not in st.session_state:
    st.session_state["gradcam_image"] = None

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

def prediction_entropy(probs):
    probs = np.clip(probs, 1e-9, 1)
    return -np.sum(probs * np.log(probs))

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

# ---------------- GRAD-CAM ----------------
def compute_gradcam(model, img_array, class_index):
    """
    Grad-CAM for a Keras model without hardcoding layer names.
    """
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # Dynamically find last Conv2D layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in model.")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-9

    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size)

    heatmap = np.array(heatmap)
    colormap = plt.get_cmap("jet")
    colored = colormap(heatmap / 255.0)
    colored = np.uint8(colored[:, :, :3] * 255)

    overlay = np.array(image) * (1 - alpha) + colored * alpha
    return Image.fromarray(np.uint8(overlay))


# ---------------- CSS ----------------
st.markdown("""
<style>
h1 {font-size:42px;color:#2E8B57;font-family:Arial;}
.card {background-color:#ffffff;border-radius:12px;padding:20px;
box-shadow:0 6px 18px rgba(0,0,0,0.07);}
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
def generate_pdf_report(
    image,
    image_size,
    top3_details,
    agreement_text,
    inference_times,
    entropy_scores,
    gradcam_image=None
):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Egyptian Vegetable Classification Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Visual Explanation</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    img_table = []

    # Original image
    orig_buf = io.BytesIO()
    image.save(orig_buf, format="PNG")
    orig_buf.seek(0)
    orig_img = RLImage(orig_buf, width=7*cm, height=7*cm)

    if gradcam_image is not None:
        cam_buf = io.BytesIO()
        gradcam_image.save(cam_buf, format="PNG")
        cam_buf.seek(0)
        cam_img = RLImage(cam_buf, width=7*cm, height=7*cm)

        img_table.append([
            Paragraph("<b>Original</b>", styles["Normal"]),
            Paragraph("<b>Grad-CAM</b>", styles["Normal"])
        ])
        img_table.append([orig_img, cam_img])
    else:
        img_table.append([orig_img])

    table = Table(
        img_table,
        colWidths=[7*cm, 7*cm] if gradcam_image is not None else [7*cm]
    )

    table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))

    elements.append(table)
    elements.append(
        Paragraph(
            f"Resolution: {image_size[0]} × {image_size[1]} pixels",
            styles["Normal"]
        )
    )
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(f"Resolution: {image_size[0]} × {image_size[1]} pixels", styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Model Agreement</b>", styles["Heading2"]))
    elements.append(Paragraph(agreement_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("<b>Top Predictions</b>", styles["Heading2"]))
    for model_name, preds in top3_details.items():
        elements.append(Paragraph(
            f"<b>{model_name}</b> | "
            f"Time: {inference_times[model_name]:.3f}s | "
            f"Entropy: {entropy_scores[model_name]:.3f}",
            styles["Heading3"]
        ))
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
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- MAIN APP ----------------
st.title("🥬 Egyptian Vegetable Classifier")
st.markdown("*Upload a vegetable → Get authentic Egyptian recipes & nutrition!*")

with st.expander("ℹ️ Model Scope & Limitations"):
    st.markdown("""
    - Trained on **15 vegetable classes only**
    - Uses **closed-set classification**
    - Out-of-distribution inputs (e.g. fruits, objects, people) are **rejected**
    - Predictions are **confidence-gated**
    - Low-confidence or conflicting model outputs are labeled as **Unknown**
    """)

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
    confs = [conf*100 for _, conf in preds]
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
    
# ---------------- INPUT QUALITY CHECKS ----------------
def is_image_blurry(image, threshold=100.0):
    """Returns True if image is blurry based on Laplacian variance."""
    gray = np.array(image.convert("L"))
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold, lap_var

MIN_WIDTH, MIN_HEIGHT = 100, 100  # Minimum acceptable resolution

if image_input:
    # Resolution check
    if image_input.width < MIN_WIDTH or image_input.height < MIN_HEIGHT:
        st.error("❌ Image resolution too low for reliable classification.")
        st.stop()

    # Blur check
    import cv2
    blurry, lap_var = is_image_blurry(image_input)
    if blurry:
        st.error(f"❌ Image is too blurry (Laplacian variance={lap_var:.2f}).")
        st.stop()

    st.info("✅ Input quality checks passed. We validate input quality before inference.")


# ---------------- PREDICTION LOGIC ----------------
if image_input:
    resized = image_input.resize(IMG_SIZE)
    cnn_batch = np.expand_dims(np.array(resized),0)
    eff_batch = np.expand_dims(np.array(resized).astype(np.float32), 0)

    top3_details = {}
    inference_times = {}
    entropy_scores = {}
    best_predictions = {}

    if model_choice in ["Custom CNN","Compare Both"] and cnn_model:
        start=time.time()
        preds=cnn_model.predict(cnn_batch,verbose=0)[0]
        inference_times["Custom CNN"]=time.time()-start
        entropy_scores["Custom CNN"] = prediction_entropy(preds)
        top3=get_top_predictions(preds)
        top3_details["Custom CNN"]=top3
        best_predictions["Custom CNN"]=top3[0][0]
        top_idx = np.argmax(preds)
        confidence = preds[top_idx]
        predicted_class = CLASS_NAMES[top_idx]

    if model_choice in ["EfficientNetB0","Compare Both"] and effnet_model:
        start=time.time()
        preds=effnet_model.predict(eff_batch,verbose=0)[0]
        inference_times["EfficientNetB0"]=time.time()-start
        entropy_scores["EfficientNetB0"] = prediction_entropy(preds)
        top3=get_top_predictions(preds)
        top3_details["EfficientNetB0"]=top3
        best_predictions["EfficientNetB0"]=top3[0][0]
        top_idx = np.argmax(preds)
        confidence = preds[top_idx]
        predicted_class = CLASS_NAMES[top_idx]

    # Determine final class
    reasons = []
    if len(best_predictions)==2:
        if best_predictions["Custom CNN"]==best_predictions["EfficientNetB0"]:
            final_class=best_predictions["Custom CNN"]
            final_conf=max(d[0][1] for d in top3_details.values())
            agreement="🟢 Both models agree!"
        else:
            final_class=None
            final_conf=0
            agreement="🟡 Models disagree"
            reasons.append("Model disagreement")
    else:
        final_class=list(best_predictions.values())[0] if best_predictions else None
        final_conf=list(top3_details.values())[0][0][1] if top3_details else 0
        agreement="Single model result"

    if final_conf < CONFIDENCE_THRESHOLD:
        reasons.append("Low confidence")

    is_unknown = final_class is None or len(reasons) > 0
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
            st.caption(f"Entropy: {entropy_scores[model_name]:.3f}")
            plot_top_predictions(model_name,preds)
        st.markdown('</div>',unsafe_allow_html=True)
        if final_class==UNKNOWN_LABEL:
            st.warning("⚠️ Prediction rejected due to low confidence or model disagreement.")
    # ---------------- GRAD-CAM VISUALIZATION ----------------
    if (
        final_class != UNKNOWN_LABEL
        and model_choice in ["EfficientNetB0", "Compare Both"]
        and effnet_model is not None
        and final_conf >= HIGH_CONF_THRESHOLD
        and agreement.startswith("🟢")
        ):
        st.markdown("### 🔍 Model Attention (Grad-CAM)")
        st.markdown("*This visualization shows which regions influenced the model’s decision.*")

        heatmap = compute_gradcam(effnet_model, eff_batch, top_idx)
        gradcam_image = overlay_gradcam(image_input, heatmap)
        st.session_state["gradcam_image"] = gradcam_image

        st.image(
            gradcam_image,
            caption="Grad-CAM: Model Focus Regions",
            use_column_width=True
            )

        st.info(
            "🧠 Grad-CAM highlights regions that most influenced the model’s confident prediction."
            )
    else:
        st.warning(
            "Grad-CAM suppressed due to low confidence, model disagreement, "
            "or unsupported input."
            )


    # PDF Download
    if top3_details:
        pdf_buffer = generate_pdf_report(
    image_input,
    image_input.size,
    top3_details,
    agreement,
    inference_times,
    entropy_scores,
    gradcam_image=st.session_state.get("gradcam_image")
)

        st.download_button(
            "📄 Download Egyptian Recipe Report (PDF)",
            data=pdf_buffer,
            file_name=f"egyptian_vegetable_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
            mime="application/pdf"
        )

else:
    st.info("👆 Upload a vegetable image or take a photo to discover authentic Egyptian recipes!")
    st.markdown("**Supported vegetables:** " + ", ".join(CLASS_NAMES))



