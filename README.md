# Document Verification API

This repository contains a Flask API for verifying documents using image processing and deep learning techniques. The API can process both image and PDF document uploads and determine if they are similar to documents in a specified folder within an AWS S3 bucket.

## Prerequisites

- Python 3.x
- Required Python packages can be installed using `pip`:

- Set up AWS credentials (Access Key ID and Secret Access Key) to access the S3 bucket.

## File Structure

Here's the file structure of the project:

- `api.py`: The main Flask API script.
- `aws_util.py`: Utility functions for interacting with AWS S3.
- `docnet.h5`: Pre-trained model file.
- `model_util.py`: Utility functions for image processing and model usage.
- `requirements.txt`: List of required Python packages.
- `Dockerfile`: Configuration for creating a Docker container.
- `flask_util.py`: Additional utility functions for Flask.
- `README.md`: This file providing project information and instructions.
- `sample_doc/`: A directory containing sample image and PDF files for testing.
  - `imagejpg`: Example image file for testing.
  - `pdf.pdf`: Example PDF file for testing.




## Getting Started

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/docnet.git
cd docnet
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. AWS Access Keys and Secret
create an environment file called `.env` and configure the following variables:

```
key=xxx
secret=xxxx
```

- `key`: Replace xxx with the AWS_ACCESS_KEY_ID.
- `secret`: Replace xxxx with the AWS_SECRET_ACCESS_KEY .


4. Open `api.py` and configure the following variables:

- `BUCKET_NAME`: Replace with the name of your AWS S3 bucket.

5. Run the Flask API:

```
python api.py
```

The API will start running on `http://127.0.0.1:5000`.

## Docker
1. Install Docker.

2. Build the Docker image
```
docker build -t docnet .
```

3. Run a Docker container from the built image
```
docker run -p 5000:5000 docnet
```
The API will start running on `http://127.0.0.1:5000`.

## How to Use

1. Make a POST request to the `/verify_document/<s3_bucket_folder>` endpoint with the `document` file part containing the document to be verified:

```
curl -X POST -F "document=@path/to/document.jpg" http://127.0.0.1:5000/verify_document/456789
```

Replace `path/to/document.jpg` with the path to the image or PDF document you want to verify.

2. The API will process the uploaded document. The response will indicate if the document is verified or not.

## Model
You can download the pre-trained `docnet.h5` file from [link] (https://drive.google.com/drive/folders/1nbgsuMHn3TuMsEKdRHFPlhhnaqfJCw9s?usp=sharing).

## Notes

- Make sure to replace placeholders such as AWS credentials, bucket name, s3 folder name, and other settings with your actual values.
- The `model_util.py` and `aws_util.py` modules contain utility functions for model processing and AWS S3 interactions.

## License

This project is licensed under the MIT License.







