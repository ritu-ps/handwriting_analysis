import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytesseract
import joblib
import pandas as pd

# Set page config
st.set_page_config(page_title="Handwriting Analysis", page_icon="✍️", layout="wide")

# Title and description
st.title("✍️ Handwriting Personality & Mood Analysis")
st.markdown("""
Upload an image of handwritten text to analyze personality traits and mood based on handwriting features.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app analyzes handwriting to predict personality traits and mood using:
    - Image processing (binarization, segmentation)
    - Feature extraction (slant, size, spacing, etc.)
    - Machine learning models
    """)

    st.header("How to Use")
    st.markdown("""
    1. Upload a clear image of handwriting
    2. Preprocess the image if needed
    3. View extracted features
    4. See personality predictions
    """)

# Initialize session state
if 'features' not in st.session_state:
    st.session_state.features = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Image preprocessing
def preprocess_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return processed

# Feature extraction from image
def extract_features(image_path, preprocessed_img):
    features = {}
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ocr_data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    word_heights = [h for h in ocr_data['height'] if h > 0]
    features['avg_word_height'] = np.mean(word_heights) if word_heights else 0

    if len(ocr_data['text']) > 1:
        spaces = []
        for i in range(1, len(ocr_data['text'])):
            if ocr_data['text'][i - 1] and ocr_data['text'][i]:
                spaces.append(ocr_data['left'][i] - (ocr_data['left'][i - 1] + ocr_data['width'][i - 1]))
        features['avg_word_spacing'] = np.mean(spaces) if spaces else 0
    else:
        features['avg_word_spacing'] = 0

    if len(ocr_data['text']) > 1:
        line_spaces = []
        prev_top = None
        for i in range(len(ocr_data['text'])):
            if ocr_data['text'][i]:
                if prev_top is not None:
                    line_spaces.append(ocr_data['top'][i] - prev_top)
                prev_top = ocr_data['top'][i]
        features['avg_line_spacing'] = np.mean(line_spaces) if line_spaces else 0
    else:
        features['avg_line_spacing'] = 0

    edges = cv2.Canny(preprocessed_img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 45:
                angles.append(angle)
        features['avg_slant'] = np.mean(angles) if angles else 0
    else:
        features['avg_slant'] = 0

    contours, _ = cv2.findContours(preprocessed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(cnt) for cnt in contours]
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]
        stroke_widths = [area / perimeter if perimeter > 0 else 0 for area, perimeter in zip(areas, perimeters)]
        features['avg_stroke_width'] = np.mean(stroke_widths) if stroke_widths else 0
    else:
        features['avg_stroke_width'] = 0

    if len(ocr_data['text']) > 0 and ocr_data['left'][0] > 0:
        features['left_margin'] = ocr_data['left'][0]
    else:
        features['left_margin'] = 0

    n_components = cv2.connectedComponentsWithStats(preprocessed_img, 4, cv2.CV_32S)
    features['connected_components'] = n_components[0]

    return features

# ML-based personality & mood prediction
def predict_personality(features):
    import numpy as np
    if 'mood_model' not in st.session_state:
        st.session_state.mood_model = joblib.load("mood_model.pkl")
        st.session_state.trait_model = joblib.load("personality_model.pkl")
        st.session_state.mood_encoder = joblib.load("mood_encoder.pkl")
        st.session_state.trait_encoder = joblib.load("personality_encoder.pkl")

    try:
        input_vector = np.array([[
            features['avg_word_height'],
            features['avg_word_spacing'],
            features['avg_line_spacing'],
            features['avg_slant'],
            features['avg_stroke_width'],
            features['left_margin'],
            features['connected_components']
        ]])
    except KeyError as e:
        st.error(f"Missing feature: {e}")
        return {}

    mood_pred = st.session_state.mood_model.predict(input_vector)[0]
    trait_pred = st.session_state.trait_model.predict(input_vector)[0]

    mood_label = st.session_state.mood_encoder.inverse_transform([mood_pred])[0]
    trait_label = st.session_state.trait_encoder.inverse_transform([trait_pred])[0]

    return {
        "Predicted Personality Trait": trait_label,
        "Predicted Mood": mood_label
    }

# Upload & analyze image
uploaded_file = st.file_uploader("Upload a handwriting sample", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    processed_img = preprocess_image(image)

    with col2:
        st.subheader("Processed Image")
        st.image(processed_img, use_container_width=True, clamp=True)

    if st.button("Analyze Handwriting"):
        with st.spinner("Extracting features..."):
            features = extract_features("temp_image.jpg", processed_img)
            st.session_state.features = features

        with st.spinner("Predicting personality & mood..."):
            predictions = predict_personality(features)
            st.session_state.predictions = predictions

        st.subheader("Extracted Features")
        features_df = pd.DataFrame.from_dict(features, orient='index', columns=['Value'])
        st.dataframe(features_df.style.format("{:.2f}"))

        st.subheader("Predicted Mood and Personality")
        for key, value in predictions.items():
            st.markdown(f"**{key}**: {value}")

        st.subheader("Feature Visualization")
        viz_features = {
            'Word Height': features['avg_word_height'],
            'Word Spacing': features['avg_word_spacing'],
            'Line Spacing': features['avg_line_spacing'],
            'Slant Angle': features['avg_slant'],
            'Stroke Width': features['avg_stroke_width']
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(viz_features.keys(), viz_features.values())
        ax.set_title("Handwriting Feature Comparison")
        ax.set_ylabel("Value")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        with st.expander("How to interpret these results"):
            st.markdown("""
            - **Word Height**: Larger writing suggests outgoing personality, smaller writing indicates detail-orientation
            - **Slant**: Right slant shows emotional responsiveness, left slant suggests independence
            - **Pressure**: Heavy pressure indicates high energy, light pressure suggests calmness
            - **Spacing**: Wide spacing shows logical thinking, narrow spacing indicates creativity
            - **Margins**: Large left margins suggest traditional approach, small margins show spontaneity
            """)

