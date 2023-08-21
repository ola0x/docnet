import os
from flask import Flask, request, jsonify
from PIL import Image
import io
from model_util import DeepModel
from aws_util import process_img, process_images_in_specific_bucket_folder
import fitz

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [DeepModel.findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

def process_document(file_stream):
    file_extension = file_stream.filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:
        image_data = file_stream.read()
        # image = Image.open(io.BytesIO(image_data))
        processed_image = process_img(image_data)
        return [processed_image]  # Wrap the single image in a list
    elif file_extension == 'pdf':
        pdf_data = file_stream.read()
        pdf_document = fitz.open(stream=pdf_data, filetype='pdf')
        images = []
        for page_number in range(pdf_document.page_count):
            pdf_page = pdf_document[page_number]
            image_list = pdf_page.get_pixmapmatrix().get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                image = pdf_document.extract_image(xref)
                image_data = image["image"]
                image_extension = image["ext"]
                image = Image.open(io.BytesIO(image_data))
                processed_image = DeepModel.process_img(image)
                images.append(processed_image)
        return images
    else:
        return None