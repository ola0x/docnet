import os
import numpy as np
import streamlit as st
from keras import Model
from keras import Sequential
from util import process_img_path
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation

model_path = '../docnet.h5'

# Build and Load model
@st.cache_resource
def load_model():
    global model
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    model.load_weights(model_path)

    return model

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

@st.cache_data
def process_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = process_img_path(image_path)
            images.append(img)
    return np.vstack(images)

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

if __name__ == "__main__":

    db_paths = [['waec\\wa2.jpg', 'waec\\wa1.png']]  # List of image paths, each sublist contains image paths for a folder

    model = load_model()
    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    db_representations = []
    
    for folder_images in db_paths:
        folder_representations = []
        for image_path in folder_images:
            img_representation = vgg_face_descriptor.predict(process_img(image_path))[0, :]
            folder_representations.append(img_representation)
        db_representations.append(folder_representations)
    
    img1_representation = vgg_face_descriptor.predict(process_img('wat.jpg'))[0, :]
    
    similar_indices = []
    
    for folder_index, folder_representations in enumerate(db_representations):
        similar_indices_folder = find_similar_images(img1_representation, folder_representations)
        similar_indices.extend([(folder_index, idx) for idx in similar_indices_folder])
    
    if len(similar_indices) > 0:
        print("The new image is verified")
    else:
        print("The new image is not verified")