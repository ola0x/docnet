# Dashboard

This repository contains the dashboard code for verifying documents using image processing and deep learning techniques.

## Prerequisites

- Python 3.x
- Required Python packages can be installed using `pip`:

- Set up AWS credentials (Access Key ID and Secret Access Key) to access the S3 bucket.

## File Structure

Here's the file structure of the project:

- `demo.py`: The main dashboard script.
- `auto_doc_p.py`: The auto document script.
- `util.py`: Utility functions for image processing and model usage.
- `requirements.txt`: List of required Python packages.
- `README.md`: This file providing project information and instructions.
- `../docnet.h5`: Docnet model file.
- `demo-f/`: A directory containing sample image and PDF files for testing.
  - `imagejpg`: Example image file for testing.
  - `pdf.pdf`: Example PDF file for testing.

## DashBoard Image

![image of the dashboard](https://github.com/ola0x/docnet/blob/main/dashboard/demo-f/web-dashboard.JPG)


## Getting Started

1. Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/docnet.git
cd docnet/dashboard
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Open `auto_doc_p.py` and replace `model_path` with the name model_path variables:

  Open the `util.py` and replace the `image_folder` with your demo-folder path

4. Run the Streamlit APP:

```
streamlit run demo.py
```

The APP will start running on `http://127.0.0.1:8501`.

## Model
You can download the pre-trained `docnet.h5` file from [link] (https://drive.google.com/drive/folders/1nbgsuMHn3TuMsEKdRHFPlhhnaqfJCw9s?usp=sharing).

## License

This project is licensed under the MIT License.







