import os
from flask import Flask, request, jsonify
from PIL import Image
import io
from model_util import DeepModel
from aws_util import process_images_in_bucket
import fitz  # PyMuPDF

BUCKET_NAME = 'docnet-peapi'

app = Flask(__name__)

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [DeepModel.findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

def process_document(file_stream):
    file_extension = file_stream.filename.split('.')[-1].lower()

    if file_extension in ['jpg', 'jpeg', 'png']:
        image_data = file_stream.read()
        image = Image.open(io.BytesIO(image_data))
        processed_image = DeepModel.process_img(image)
        return processed_image
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

@app.route('/verify_document', methods=['POST'])
def verify_document():
    if 'document' not in request.files:
        return jsonify({'error': 'No document file part'})

    document = request.files['document']

    processed_images = process_images_in_bucket(BUCKET_NAME)
    uploaded_images = process_document(document)

    if uploaded_images is None:
        return jsonify({'error': 'Unsupported file format'})

    model = DeepModel._define_model()
    db_representations = model.predict(processed_images)

    result = []

    for img1_representation in uploaded_images:
        similar_indices = find_similar_images(img1_representation, db_representations)
        if len(similar_indices) > 0:
            result.append('The new doc is verified')
        else:
            result.append('The doc is not verified')

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
