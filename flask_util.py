import io
import fitz
from PIL import Image
from model_util import DeepModel
from aws_util import process_img

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
            image_list = pdf_page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                image = pdf_document.extract_image(xref)
                image_data = image["image"]
                image_extension = image["ext"]
                processed_image = process_img(image_data)
                images.append(processed_image)
        return images
    else:
        return None