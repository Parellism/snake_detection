import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from PIL import Image
from tensorflow.keras.models import load_model


st.title('Snake Detection')
st.text('Upload Image')

model = load_model('model/keras_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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
		class_names = ['non-venomous', 'venomous']
		predicted_class = class_names[y_out[0]]
		plt.imshow(img_resized[0])
		plt.show()
		st.text(f'This snake is {predicted_class}')
                
# #Split data into Training and testing
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(flat_data,target,test_size=0.3,random_state=109)

# # Compile the model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# from sklearn.model_selection import GridSearchCV
# from sklearn import svm
# param_grid = [
#               {'C':[1,10,100,1000],'kernel':['linear']},
#               {'C':[1,10,100,1000],'gamma':[0.001,0.0001],'kernel':['rbf']},    
# ]
# svc = svm.SVC(probability=True)
# clf = GridSearchCV(svc,param_grid)
# clf.fit(x_train,y_train)

# y_pred = clf.predict(x_test)

# from sklearn.metrics import accuracy_score,confusion_matrix
# accuracy_score(y_pred,y_test)
# confusion_matrix(y_pred,y_test)
