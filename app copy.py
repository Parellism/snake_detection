import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import shutil

import os
import random
import webbrowser

#=================================== Title ===============================
st.title("""
Berbisa atau tidak ?
	""")

#================================= Title Image ===========================
st.text("""""")

#======================== Time To See The Magic ===========================
st.write("""
## 👁️‍🗨️ Time To See The Magic 🌀
	""")

#========================== File Uploader ===================================
img_file_buffer = st.file_uploader("Upload an image here 👇🏻")

try:
    image = Image.open(img_file_buffer)
    img_array = np.array(image)
    st.write("""
        Preview 👀 Of Given Image!
        """)
    if image is not None:
        st.image(
            image,
            use_column_width=True
        )
    st.write("""
        Now, you are just one step ahead of prediction.
        """)
    st.write("""
        **Just Click The '👉🏼 Predict' Button To See The Prediction Corresponding To This Image! 😄**
        """)
except:
    st.write("""
        ### ❗ Any Picture hasn't selected yet!!!
        """)

#================================= Predict Button ============================
st.text("""""")
submit = st.button("👉🏼 Predict")

#==================================== Model ==================================
def processing(testing_image_path):
    IMG_SIZE = 50
    img = load_img(testing_image_path,
                   target_size=(IMG_SIZE, IMG_SIZE), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape((1, 50, 50, 1))
    prediction = model.predict(img_array)
    return prediction


def generate_result(prediction):
    class_names = ['Non Venomous', 'Venomous']
    prediction = class_names[np.argmax(prediction)]
    return prediction


#=========================== Predict Button Clicked ==========================
if submit:
    try:
        # save image on that directory
        if not os.path.exists("temp_dir"):
            os.makedirs("temp_dir")

        save_img("temp_dir/test_image.png", img_array)

        image_path = "temp_dir/test_image.png"
        # Predicting
        st.write("👁️ Predicting...")

        model_path_h5 = "model/model.h5"
        model = tensorflow.keras.models.load_model(model_path_h5)

        prediction = processing(image_path)
        result = generate_result(prediction)

        st.write(f"Prediction: {result}")
        print(f'PREDICTED OUTPUT: {prediction}')

    except:
        st.write("""
        ### ❗ Oops... Something Is Going Wrong
        """)

#=============================== Copy Right ==============================
st.text("""""")
st.text("""""")
st.text("""""")
st.write("""
### ©️ Created By awokwaokaowkwao
    """)
