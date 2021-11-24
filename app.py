import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.header("Cat/Dog Classfier")
st.write("This application can be used to classify images whether a cat or a dog is depicted on them.")

data = st.file_uploader(label="Bild hochladen")

try:
    st.image(data)
except:
    st.warning("No image uploaded!")
    st.stop()

if st.button(label = "Classify"):
    data = Image.open(data)
    image = data.resize((224, 224), Image.LANCZOS)
    #st.image(image)

    # from tensorflow.keras.applications.vgg16 import VGG16
    # vgg16_model = VGG16(include_top=False, input_shape=(224, 224, 3))

    image = np.asarray(image)
    print(image.shape)

    data = image.reshape(1, 224,224,3)
    print(data.shape)

    # X_after_vgg = vgg16_model.predict(data)

    from tensorflow.keras.models import load_model
    model = load_model("TransferlearningMNV2.h5")
    prediction = model.predict(data).astype(np.float32)
    proba = model.predict(data)

    if prediction == 1:
        st.subheader(f"It is to {proba[0][0]} a cat! 🐈")
    else:
        st.subheader(f"It is to {proba[0][0]} a dog! 🐕 ")