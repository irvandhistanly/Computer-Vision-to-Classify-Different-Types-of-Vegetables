# prediction_irvandhi.py

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model_improved.keras')

model = load_model()

# Define class labels
class_labels = ['Bean', 'Bitter Gourd', 'Bottle Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# Preprocess the input image
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))  # Resize the image to match the input size of the model
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img

# Get the predicted class label
def get_predicted_class(prediction):
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_class_index]
    return predicted_class

# Main prediction function
def main():
    st.title('Model Prediction')

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Perform inference
        preprocessed_img = preprocess_image(img)
        input_img = np.expand_dims(preprocessed_img, axis=0)  # Add batch dimension
        prediction = model.predict(input_img)
        predicted_class = get_predicted_class(prediction)

        # Display the image and prediction
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.success(f'Predicted class: {predicted_class}')

# Run the app
if __name__ == '__main__':
    main()