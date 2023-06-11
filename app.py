import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from tensorflow.keras.models import load_model

page_bg_img = """
<style>
[data-testid-"stAppViewContainer"] {
background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQesuS4zrN__rOO_oQvAuTV2yeLZAjiVs2GXA&usqp=CAU");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('Snake Detection')
st.text('Upload Image')

model = load_model('model/keras_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
class_names = ['non-venomous', 'venomous']

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	img = Image.open(uploaded_file)
	st.image(img,caption='Uploaded Image')

	if st.button('PREDICT'):
		img_array = np.array(img)
		img_resized = img.resize((224, 224))
		img_resized = np.expand_dims(img_resized, axis=0)
		y_out = model.predict(img_resized)
		y_out = np.argmax(y_out, axis=1)
		predicted_class = class_names[y_out[0]]
		plt.imshow(img_resized[0])
		plt.show()
		st.text(f'This snake is {predicted_class}')
               
