import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

# Load the trained models with corrected paths
model1 = load_model(r"D:\R\pycharm\PyCharm Community Edition 2024.1.4\project\cotton disease\model_resnet152V2.h5")
model2 = load_model(r"D:\R\pycharm\PyCharm Community Edition 2024.1.4\project\cotton disease\model_inception.h5")
model3 = load_model(r"D:\R\pycharm\PyCharm Community Edition 2024.1.4\project\cotton disease\model_vgg.h5")

# Function to preprocess the uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to make predictions
def predict(img_array, model):
    prediction = model.predict(img_array)
    return "Diseased" if prediction[0][0] > 0.5 else "Healthy"

# Streamlit app
st.title("Cotton Disease Prediction")
st.write("Upload an image to find cotton disease")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    img = load_img(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(img)

    # Model selection
    model_option = st.selectbox(
        'Which model would you like to use?',
        ('Inception', 'VGG', 'ResNet')
    )

    if st.button('Predict'):
        if model_option == 'Inception':
            model = model2
            selected_model = "Inception"
        elif model_option == 'VGG':
            model = model3
            selected_model = "VGG"
        else:
            model = model1
            selected_model = "ResNet"
        
        prediction = predict(img_array, model)
        st.write(f'Selected Model: {selected_model}')
        st.write(f'Prediction: {prediction}')
        
        # Show the image again after prediction
        # st.image(img, caption='Processed Image.', use_column_width=True)

    # Debugging steps: Show model summary
    st.write("Model Summary:")
    model_summary_str = []
    model.summary(print_fn=lambda x: model_summary_str.append(x))
    model_summary_str = "\n".join(model_summary_str)
    st.text(model_summary_str)
