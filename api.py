import os

import numpy as np
from model_util import DeepModel

def process_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            img = DeepModel.process_img(image_path)
            images.append(img)
    return np.vstack(images)

if __name__ == "__main__":

    image_x = DeepModel.process_img('dri3.jpeg')
    image_y = DeepModel.process_img('drit.png')

    model = DeepModel._define_model()

    img1_representation = model.predict(image_x)[0,:]
    img2_representation = model.predict(image_y)[0,:]

    cos_d = DeepModel.findCosineDistance(img1_representation, img2_representation)
    ecl_d = DeepModel.findEuclideanDistance(img1_representation, img2_representation)
    print(cos_d)
    print(img1_representation)