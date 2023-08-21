# Document Verification API

This repository contains a Flask API for verifying documents using image processing and deep learning techniques. The API can process both image and PDF document uploads and determine if they are similar to documents in a specified folder within an AWS S3 bucket.

## Prerequisites

- Python 3.7.x
- Required Python packages can be installed using `pip`:

- Set up AWS credentials (Access Key ID and Secret Access Key) to access the S3 bucket.

## Getting Started

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/document-verification-api.git
cd document-verification-api
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. JWT Secret and Algorithm
create an environment file called `.env` and configure the following variables:

```
key=AKIARTU374NWES7H5TXN
secret=tERHye0GSDSn51Z0ENwBw3hB7Swx8O3Y85MN0H6x
```

- `key`: Replace with the AWS_ACCESS_KEY_ID.
- `secret`: Replace with the AWS_SECRET_ACCESS_KEY .


4. Open `app.py` and configure the following variables:

- `BUCKET_NAME`: Replace with the name of your AWS S3 bucket.

5. Run the Flask API:

```
python app.py
```


The API will start running on `http://127.0.0.1:5000`.

## How to Use

1. Make a POST request to the `/verify_document/<s3_bucket_folder>` endpoint with the `document` file part containing the document to be verified:

```
curl -X POST -F "document=@path/to/document.jpg" http://127.0.0.1:5000/verify_document/456789
```

Replace `path/to/document.jpg` with the path to the image or PDF document you want to verify.

2. The API will process the uploaded document and determine if it's similar to documents in the specified folder. The response will indicate if the document is verified or not.

## Notes

- Make sure to replace placeholders such as AWS credentials, bucket name, specific folder name, and other settings with your actual values.
- This is a basic setup. For production use, consider implementing security measures, error handling, and using a production-ready server.
- The `model_util.py` and `aws_util.py` modules contain utility functions for model processing and AWS S3 interactions. Make sure to include or adapt them according to your project's structure.

## License

This project is licensed under the MIT License.







