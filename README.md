# Drawing Recognition

Web app for interactive drawing and drawing recogintion using custom Cnn_model trained on [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)

## Contents
- [Info](#Info)
- [Tech](#Tech)
- [Installation & Running Locally](#Installation & Running Locally)
- [Creating new models](#Creating new models)

## Info

This web application recognizes hand-drawn sketches using a machine learning model based on a convolutional neural network (CNN), implemented with PyTorch. The application processes user drawings, classifies them, and returns predictions with confidence scores.

The CNN model was trained with preprocessed datasets to ensure high classification accuracy. After tuning various parameters, the final model achieved:

92.55% accuracy on the validation dataset
92.94% accuracy on the test dataset

The web interface allows users to draw directly on a digital canvas, submit their drawings, and receive classification results instantly. The backend efficiently processes and classifies sketches.

Final tests demonstrated 96% accuracy on real user-drawn samples, confirming the application’s effectiveness.

## Tech

Backend
- Python 3.11.4
- PyTorch 2.3.1+cu118
- Torchvision 0.18.1+cu118
- NumPy 1.26.4
- Matplotlib 3.8.4
- Flask 3.0.3
- Flask-CORS 5.0.0
- Cairocffi 1.7.1
- Scikit-learn

Frontend
- HTML5
- JavaScript

## Installation & Running Locally

Step 1: Clone the Repository
``` bash
git clone <url>
cd project-root
```
Step 2: Set Up a Virtual Environment (Recommended)
``` bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```
Step 3: Install Dependencies
``` bash
pip install -r requirements.txt
```
Step 4: Run the Backend
Start the Flask server:
``` bash
cd backend
python app.py
```
This will start the backend at http://127.0.0.1:5000/.

Step 6: Run the Frontend
- The frontend is a static HTML/JS page, so you can simply open frontend/site.html in a browser.

## Creating new models
  If you want to edit existing model or train model on new data you are welcome to do so

  Step 1: Downland the Simplified drawings files from [quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)

  Step 2: Prepare the configuration.json file, it should look like this:
  
```json
{
    "trainig data path": "C:/path_were_you_want_to_have_training_data",
    "validation data path": "C:/path_were_you_want_to_have_validation_data",
    "test data path": "C:/path_were_you_want_to_have_test_data",
    "raw data path": "C:/path_were_you_want_to_have_numpy_data",
    "raw ndjson data path": "C:/path_were_you_have_dowlanded_nsjdon_data"
}
```
Step 3: Change data from ndjson format to .npy format using **`Change_From_Ndjson_To_Numpy.py`**

Step 4: Separte data to train/val/test sets using **`Separate_data_into_sets.py`**

Step 5: (Optional) Change or edit model architecture in **`model/cnn_model.py`** if you do remember to also change it in app in **`backend/model/cnn_model.py`**

Step 6: Train and save model using **`Create_cnc_model.py`**
