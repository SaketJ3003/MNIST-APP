# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# import cv2
# import streamlit_drawable_canvas as draw
#
# # Load the trained model
# @st.cache_resource
# def load_model():
#     return keras.models.load_model("mnist_model.h5")
#
# model = load_model()
#
# # Streamlit UI
# st.title("MNIST Handwritten Digit Recognizer")
# st.write("Draw a digit (0-9) in the box below, and the model will predict it.")
#
# # Create a drawing canvas
# canvas_result = draw.st_canvas(
#     fill_color="black",  # Background color
#     stroke_width=10,
#     stroke_color="white",  # Drawing color
#     background_color="black",  # Canvas background
#     width=280,
#     height=280,
#     drawing_mode="freedraw",
#     key="canvas"
# )
#
# if canvas_result.image_data is not None:
#     image = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
#     image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
#     image = cv2.bitwise_not(image)  # Invert colors
#     image = image / 255.0  # Normalize
#     image = image.reshape(1, 28, 28, 1)
#
#     # Make prediction
#     prediction = model.predict(image)
#     predicted_digit = np.argmax(prediction)
#
#     # Display results
#     st.write(f"Prediction: {predicted_digit}")
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import streamlit_drawable_canvas as draw
import time

# MUST be the first Streamlit call in the script:
st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✍️", layout="centered")

# Load the trained model (safe after set_page_config)
@st.cache_resource
def load_model():
    return keras.models.load_model("mnist_model.h5")

# call it after set_page_config so the cache_resource doesn't run before config
model = load_model()

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: #ff6600;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #444;
        }
        .stButton>button {
            background: linear-gradient(135deg, #ff6600, #ff3300);
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #ff3300, #cc0000);
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div class='title'>🖌️ MNIST Handwritten Digit Recognizer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Draw a digit (0-9) below and let the AI predict it!</div>", unsafe_allow_html=True)

# Create a drawing canvas
canvas_result = draw.st_canvas(
    fill_color="black",  # Background color
    stroke_width=10,
    stroke_color="white",  # Drawing color
    background_color="black",  # Canvas background
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

st.write("\n")
if canvas_result is not None and canvas_result.image_data is not None:
    if st.button("🔮 Predict Digit", key="predict_button"):
        with st.spinner("✨ Analyzing your drawing... Please wait ✨"):
            time.sleep(1)
            image = cv2.resize(canvas_result.image_data.astype("uint8"), (28, 28))
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            image = cv2.bitwise_not(image)  # Invert colors
            image = image / 255.0  # Normalize
            image = image.reshape(1, 28, 28, 1)

            # Make prediction
            prediction = model.predict(image)
            predicted_digit = np.argmax(prediction)

            # Display results with animation
            st.success("✅ Prediction complete! 🎉")
            st.markdown(f"<h2 style='text-align: center; color: #ff3300; font-size: 36px;'>Predicted Digit: {predicted_digit}</h2>", unsafe_allow_html=True)
            st.balloons()
