import io
import fitz
from model_util import DeepModel, fs_images
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from aws_util import process_images_in_specific_bucket_folder
from flask_util import process_document

BUCKET_NAME = 'local-periscope-bucket'

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)

ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'png'}

# Initialize the model
model = DeepModel._define_model()

@app.route('/healthz', methods = ['GET'])
def health():
    return jsonify(
      application='Doc Verification',
      version='1.0.0',
      message= "endpoint working..."
    )

@app.route('/verify_document', methods=['POST'])
def verify_document():
    
    if 'document' not in request.files:
        return jsonify({'error': 'No document file part'})

    if request.form['processedbucket'] is None:
        return jsonify({'error': 'Specify path to processed bucket'})

    document = request.files['document']

    processed_images = process_images_in_specific_bucket_folder(BUCKET_NAME, specific_folder=request.form['processedbucket'])
    uploaded_images = process_document(document)

    if uploaded_images is None:
        return jsonify({'error': 'Unsupported file format'})

    db_representations = model.predict(processed_images)

    result = []

    for img1_representation in uploaded_images:
        img1_representation = model.predict(img1_representation)[0,:]
        similar_indices = fs_images(img1_representation, db_representations)
        if len(similar_indices) > 0:
            result.append('Verified')
        else:
            result.append('Not verified')
        
    return jsonify({
        "message": "successful",
        'result': result
    })
        
if __name__ == "__main__":
    print("Starting Doc verification...")
    app.run()