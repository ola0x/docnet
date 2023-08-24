import io
import glob
import numpy as np
import streamlit as st
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def process_img(image_bytes):
    img = load_image_from_bytes(image_bytes)
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@st.cache_data
def process_img_path(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def load_images(image_folder):
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif"]

    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(f"{image_folder}/{ext}"))
    manuscripts = []
    for image_file in image_files:
        image_file = image_file.replace("\\", "/")
        parts = image_file.split("/")
        if parts[1] not in manuscripts:
            manuscripts.append(parts[1])
    manuscripts.sort()

    return image_files, manuscripts