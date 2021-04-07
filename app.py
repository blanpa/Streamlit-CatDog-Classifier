import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.header("CatDog Classfier")
st.write("Mit dieser Anwendung können Bilder klassifiziert werden, ob darauf eine Katze oder ein Hund abgebildet wird.")

data = st.file_uploader(label="Bild hochladen")

try:
    st.image(data)
except:
    st.warning("Kein Bild hochgeladen!")
    st.stop()

if st.button(label = "Klassifizieren"):
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
    proba = model.predict_proba(data)

    if prediction == 1:
        st.subheader(f"Es ist zu {proba[0][0]} eine Katze! 🐈")
    else:
        st.subheader(f"Es ist zu  {proba[0][0]} ein Hund! 🐕 ")