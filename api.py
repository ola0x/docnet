import io
import fitz
from PIL import Image
from model_util import DeepModel
from flask import Flask, request, jsonify
from aws_util import process_images_in_specific_bucket_folder
from flask_util import find_similar_images, process_document

BUCKET_NAME = 'docnet-peapi'

app = Flask(__name__)

model = DeepModel._define_model()

def find_similar_images(image_representation, db_representations, threshold=0.06):
    similarities = [DeepModel.findCosineDistance(image_representation, db_repr) for db_repr in db_representations]
    similar_indices = [i for i, similarity in enumerate(similarities) if similarity < threshold]
    return similar_indices

@app.route('/verify_document/<specific_folder>', methods=['POST'])
def verify_document(specific_folder):
    if not specific_folder:
        return jsonify({'error': 'Specific folder is required in the URL path'})
    
    if 'document' not in request.files:
        return jsonify({'error': 'No document file part'})

    document = request.files['document']

    processed_images = process_images_in_specific_bucket_folder(BUCKET_NAME, specific_folder=specific_folder)
    uploaded_images = process_document(document)

    if uploaded_images is None:
        return jsonify({'error': 'Unsupported file format'})

    db_representations = model.predict(processed_images)

    result = []

    for img1_representation in uploaded_images:
        img1_representation = model.predict(img1_representation)[0,:]
        similar_indices = find_similar_images(img1_representation, db_representations)
        if len(similar_indices) > 0:
            result.append('The new doc is verified')
        else:
            result.append('The doc is not verified')
        
    return jsonify({'result': result})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)