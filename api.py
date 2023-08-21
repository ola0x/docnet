import os

import numpy as np
from model_util import DeepModel
from aws_util import process_images_in_bucket

BUCKET_NAME = 'docnet-peapi'

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [DeepModel.findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

if __name__ == "__main__":

    processed_images = process_images_in_bucket(BUCKET_NAME)
    image_x = DeepModel.process_img('pas1_aa.jpg')

    model = DeepModel._define_model()

    # Process the images in the database folder
    db_representations = model.predict(processed_images)

    img1_representation = model.predict(image_x)[0,:]

    similar_indices = find_similar_images(img1_representation, db_representations)
    if len(similar_indices) > 0:
        print("The new image is similar to images at indices:", similar_indices)
    else:
        print("The new image is not similar to any image in the database.")








    # image_x = DeepModel.process_img('dri3.jpeg')
    # image_y = DeepModel.process_img('drit.png')

    # model = DeepModel._define_model()

    # img1_representation = model.predict(image_x)[0,:]
    # img2_representation = model.predict(image_y)[0,:]

    # cos_d = DeepModel.findCosineDistance(img1_representation, img2_representation)
    # ecl_d = DeepModel.findEuclideanDistance(img1_representation, img2_representation)
    # print(cos_d)
    # print(img1_representation)