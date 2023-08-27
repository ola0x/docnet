import io
import fitz
import boto3
import numpy as np
from PIL import Image
from decouple import config
from model_util import image, preprocess_input

AWS_ACCESS_KEY_ID = config("key")
AWS_SECRET_ACCESS_KEY = config("secret")

# Initialize the S3 client
s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

def load_image_from_bytes(image_bytes):
    return Image.open(io.BytesIO(image_bytes))

def process_img(image_bytes):
    img = load_image_from_bytes(image_bytes)
    img = img.resize((224, 224))
    img = img.convert('RGB')  # Ensure the image is in RGB format
    x = image.img_to_array(img, data_format='channels_last')
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def process_images_in_specific_bucket_folder(bucket_name, specific_folder):
    images = []
    
    # List objects in the bucket within the specific folder
    objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{specific_folder}/")
    
    for obj in objects.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Download the image from S3
            image_object = s3.get_object(Bucket=bucket_name, Key=key)
            image_data = image_object['Body'].read()
            
            # Process the image
            img = process_img(image_data)
            if img is not None:
                images.append(img)
        elif key.lower().endswith(('.pdf')):
            doc_object = s3.get_object(Bucket=bucket_name, Key=key)
            doc_data = doc_object['Body'].read()
            pdf_document = fitz.open(stream=doc_data, filetype='pdf')
            images = []
            for page_number in range(pdf_document.page_count):
                pdf_page = pdf_document[page_number]
                image_list = pdf_page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    image = pdf_document.extract_image(xref)
                    image_data = image["image"]
                    image_extension = image["ext"]
                    processed_image = process_img(image_data)
                    if processed_image is not None:
                        images.append(processed_image)
    if images:
        return np.vstack(images)
    else:
        return None