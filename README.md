# 🥬 Egyptian Vegetable Image Classification System

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14+-orange?style=for-the-badge\&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge\&logo=keras)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge\&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

An end-to-end **deep learning image classification system** for recognizing vegetables from images using a **Custom CNN** and **EfficientNetB0 Transfer Learning**.

The project combines machine learning, computer vision, model comparison, confidence analysis, Grad-CAM visualization, nutrition information, recipe recommendations, and automated PDF report generation into an interactive **Streamlit application**.

---

## 🔗 Live Demo

*https://vegetable-classifier-xnrf2uvq987m6rgg3tajde.streamlit.app/*

---

## 🌟 Overview

This project was developed to build a practical deep learning system capable of classifying vegetable images while providing additional information beyond the predicted class.

Two deep learning approaches are integrated into the application:

* **Custom Convolutional Neural Network (CNN)**
* **EfficientNetB0 with Transfer Learning**

The application compares the predictions of both models and provides additional information such as confidence scores, Top-3 predictions, nutrition facts, recipe suggestions, and visual explanations.

---

## 🚀 Features

* 🧠 **Custom CNN Classification**
* ⚡ **EfficientNetB0 Transfer Learning**
* 🔄 **Model Comparison**
* 🥕 **15 Vegetable Classes**
* 📊 **Top-3 Predictions**
* 📈 **Confidence Visualization**
* 🔍 **Prediction Entropy Analysis**
* 🔥 **Grad-CAM Visual Explanations**
* 🥗 **Nutrition Information**
* 🍲 **Egyptian Recipe Recommendations**
* 📄 **Automated PDF Report Generation**
* ⏱️ **Inference Time Measurement**
* 🖼️ **Image Upload and Processing**
* 🌐 **Interactive Streamlit Interface**
* 🚫 **Unknown / Unsupported Input Detection**

---

## 🥕 Supported Classes

The system currently recognizes **15 classes**:

```text
Bean
Bitter Gourd
Bottle Gourd
Brinjal
Broccoli
Cabbage
Capsicum
Carrot
Cauliflower
Cucumber
Papaya
Potato
Pumpkin
Radish
Tomato
```

---

## 🧠 Deep Learning Models

### 1. Custom CNN

A custom Convolutional Neural Network is used as one of the classification models.

The model processes the input image and produces class probabilities for the supported vegetable categories.

```text
Input Image
     ↓
Image Preprocessing
     ↓
Custom CNN
     ↓
Class Probabilities
     ↓
Top-3 Predictions
```

---

### 2. EfficientNetB0

The project also uses **EfficientNetB0** with transfer learning.

The EfficientNet model provides an alternative classification approach and allows the application to compare its predictions with the custom CNN.

```text
Input Image
     ↓
EfficientNet Preprocessing
     ↓
EfficientNetB0
     ↓
Class Probabilities
     ↓
Top-3 Predictions
```

---

## 🔬 Model Comparison

The application runs both models and compares their predictions.

For each model, the system can display:

* Predicted vegetable
* Confidence score
* Top-3 predictions
* Inference time
* Prediction entropy

This makes the project more than a simple image classifier because it allows users to examine how different deep learning approaches behave on the same input.

---

## 📊 Confidence & Prediction Analysis

The application uses prediction probabilities to analyze model confidence.

A confidence threshold is used to determine whether an input should be treated as a supported vegetable or classified as:

```text
Unknown (Non-Vegetable or Unsupported Input)
```

The application also calculates **prediction entropy** to provide an additional measure of uncertainty in the model's probability distribution.

---

## 🔥 Grad-CAM Explainability

The application includes **Grad-CAM** visualization to provide insight into which regions of an input image contribute to the model's prediction.

The generated visualization can display:

```text
Original Image
       +
Grad-CAM Heatmap
       ↓
Visual Model Explanation
```

This helps make the classification system more interpretable instead of treating the neural network as a complete black box.

---

## 🥗 Nutrition Information

After classification, the application provides nutritional information for the predicted vegetable.

Depending on the class, the application can display information such as:

* Calories
* Protein
* Carbohydrates
* Fiber
* Fat
* Vitamins
* Minerals
* Additional nutritional notes

---

## 🍲 Egyptian Recipe Recommendations

The application also provides recipe suggestions related to the detected vegetable.

Examples include:

```text
Bean
→ Ful Medames
→ Ta'meya
→ Besarah

Brinjal
→ Mahshi Betengan
→ Baba Ghanoush
→ Mesa'a'ah

Tomato
→ Salata Baladi
→ Shakshuka
→ Tomato Salsa
```

This extends the project from a pure classification model into a more practical interactive application.

---

## 📄 Automated PDF Reports

The application can generate a PDF report containing information about the classification result.

The report can include:

* Original image
* Grad-CAM visualization
* Image resolution
* Model agreement
* Top predictions
* Confidence percentages
* Inference time
* Prediction entropy

This provides a downloadable record of the classification analysis.

---

## 🧠 System Architecture

```text
                  ┌────────────────────┐
                  │    Input Image     │
                  └─────────┬──────────┘
                            │
                            ▼
                  ┌────────────────────┐
                  │ Image Preprocessing│
                  └─────────┬──────────┘
                            │
                  ┌─────────┴─────────┐
                  │                   │
                  ▼                   ▼
        ┌─────────────────┐  ┌─────────────────┐
        │   Custom CNN    │  │  EfficientNetB0 │
        │                 │  │ Transfer Learning│
        └────────┬────────┘  └────────┬────────┘
                 │                    │
                 └──────────┬─────────┘
                            ▼
                  ┌────────────────────┐
                  │ Prediction Analysis│
                  └─────────┬──────────┘
                            │
            ┌───────────────┼────────────────┐
            │               │                │
            ▼               ▼                ▼
      Top-3 Results    Confidence       Model Agreement
            │               │
            └───────┬───────┘
                    ▼
          ┌─────────────────────┐
          │  Additional Output  │
          ├─────────────────────┤
          │ Nutrition           │
          │ Recipes             │
          │ Grad-CAM            │
          │ PDF Report          │
          └─────────────────────┘
```

---

## 🗂️ Repository Structure

```text
vegetable-classifier/
│
├── Advanced.py
│
├── advanced-ai.ipynb
│
├── cnn_model (1).h5
│
├── efficientnetb0_finetuned.keras
│
├── requirements.txt
│
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   ├── comparison.png
│   ├── gradcam.png
│   └── report.png
│
├── docs/
│
├── sample_images/
│
├── LICENSE
└── README.md
```

> The `screenshots/`, `docs/`, and `sample_images/` folders should be added to the repository if they do not already exist.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUSSOFOSAMA/vegetable-classifier.git
cd vegetable-classifier
```

### 2. Create a Virtual Environment

Windows:

```cmd
python -m venv .venv
```

Activate it:

```cmd
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The current repository specifies dependencies including Streamlit, NumPy, Pillow, Matplotlib, TensorFlow, ReportLab, and OpenCV.

---

## ▶️ Running the Application

Launch the Streamlit application with:

```bash
python -m streamlit run Advanced.py
```

The application will open in your browser.

---

## 📸 Screenshots

Add screenshots of the main application here.

### Main Interface

```text
screenshots/home.png
```

### Vegetable Prediction

```text
screenshots/prediction.png
```

### Model Comparison

```text
screenshots/comparison.png
```

### Grad-CAM Visualization

```text
screenshots/gradcam.png
```

### Generated PDF Report

```text
screenshots/report.png
```

---

## 📈 Performance

The project compares two deep learning approaches:

| Model          | Architecture                 | Role           |
| -------------- | ---------------------------- | -------------- |
| Custom CNN     | Convolutional Neural Network | Baseline model |
| EfficientNetB0 | Transfer Learning            | Advanced model |

The application also records inference time and prediction uncertainty for each model.

> **Note:** Exact accuracy, precision, recall, F1-score, and confusion-matrix results should be added here once the final evaluation results are confirmed.

---

## 🔮 Future Improvements

* Improve dataset diversity and class balance.
* Add additional vegetable categories.
* Deploy the application publicly.
* Add automated model evaluation dashboards.
* Improve out-of-distribution detection.
* Add more advanced explainability techniques.
* Add database support for prediction history.
* Optimize the models for faster inference.
* Add mobile-friendly deployment.

---

## 🛠️ Technologies Used

* **Python**
* **TensorFlow**
* **Keras**
* **EfficientNetB0**
* **Streamlit**
* **OpenCV**
* **NumPy**
* **Pillow**
* **Matplotlib**
* **ReportLab**
