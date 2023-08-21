import os

from model_util import DeepModel
from aws_util import process_images_in_bucket

BUCKET_NAME = 'docnet-peapi'

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [DeepModel.findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

if __name__ == "__main__":

    processed_images = process_images_in_bucket(BUCKET_NAME)
    image_x = DeepModel.process_img('drit.png')

    model = DeepModel._define_model()

    # Process the documents in the aws s3 folder
    db_representations = model.predict(processed_images)

    img1_representation = model.predict(image_x)[0,:]

    similar_indices = find_similar_images(img1_representation, db_representations)
    if len(similar_indices) > 0:
        print("The new doc is verified")
    else:
        print("The doc is not verified")