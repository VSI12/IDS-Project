import base64
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import shutil

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, flash, render_template, request, redirect, url_for, jsonify
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename

app = Flask(__name__)
executor = ThreadPoolExecutor(max_workers=1)

app.config['SECRET_KEY']='supersecret'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.debug = True

route_accessed = {"upload_KNN": False, "upload_DecisionTree": False, "upload_NaiveBayes": False}

#for generating the foilders for the confusion matrices
confusion_matrix_folder = 'Confusion Matrices'
confusion_matrix_decisionTree = 'assets/Confusion Matrices/Confusion Matrices DecisionTree'
confusion_matrix_KNN = 'assets/Confusion Matrices/Confusion Matrices KNN'
confusion_matrix_NaiveBayes = 'assets/Confusion Matrices/confusion Matrices NaiveBayes'
confusion_matrix_RandomForest = 'assets/Confusion Matrices/confusion Matrices RandomForest'

if not os.path.exists(confusion_matrix_folder):
    os.makedirs(confusion_matrix_folder)

confusion_matrices_directories = [confusion_matrix_decisionTree,confusion_matrix_KNN,confusion_matrix_NaiveBayes,confusion_matrix_RandomForest]
for x in confusion_matrices_directories:
    if not os.path.exists(x):
        os.makedirs(x)


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload FIle")


@app.route("/")
def home():
    return render_template("index.html", model_url = url_for)

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/upload_KNN")
def upload_KNN():
    route_accessed["upload_KNN"] = True
    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_DecisionTree")
def upload_DecisionTree():
    route_accessed["upload_DecisionTree"]=True
    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_NaiveBayes")
def upload_NaiveBayes():
    route_accessed["upload_NaiveBayes"] = True
    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_RandomForest")
def upload_RandomForest():
    route_accessed["upload_RandomForest"] = True
    form = UploadFileForm()
    return render_template('upload.html', form=form)


@app.route('/results', methods=['POST'])
def submit():
    from ids_logic import preprocess
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    global processed_data
# Get uploaded file from request
    file = request.files.get('file')
    if not file or file.filename == '':
        return "Error: No file selected for upload", 400
    
    # Save the uploaded file uniquely to prevent conflicts
    file_path = '/assets/uploaded_datasets' + file.filename + str({timestamp})
    file.save('dataset.csv')

    try:
        # Preprocess the data (align columns, encode categorical features, etc.)
        processed_data = preprocess('dataset.csv')

        # Check if preprocessed data is empty
        if processed_data.empty:
            return "Error: Processed dataset is empty", 400

    except pd.errors.EmptyDataError:
        return "Error: Uploaded file is empty or contains no data", 400
    except Exception as e:
        return f"Error during preprocessing: {str(e)}", 500

    # Run further processing asynchronously
    # executor.submit(process, processed_data)
    print("uploaded")
        
    return render_template('result.html')

def process():
 # Generate a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    
    # processed_data = session.get('processed_data')

    if route_accessed["upload_DecisionTree"] == True:
            from ids_logic import DecisionTree
            route_accessed["upload_DecisionTree"]=False
            img_base64 = DecisionTree(processed_data)
            
            os.remove('dataset.csv')  # Remove uploaded file
            # Return data as JSON
            return jsonify({'confusion_matrix': img_base64})
            # return render_template('result.html', confusion_matrix=img_base64, results=results)

    elif route_accessed["upload_KNN"] == True:
            from ids_logic import KNN
            route_accessed["upload_KNN"]=False
            route_accessed["upload_DecisionTree"]=False
            img_base64 = KNN(processed_data)
            
            os.remove('dataset.csv')  # Remove uploaded file
            # Return data as JSON
            return jsonify({'confusion_matrix': img_base64})
            
    elif route_accessed["upload_NaiveBayes"] == True:
            from ids_logic import GaussianNB
            route_accessed["upload_KNN"]=False
            route_accessed["upload_DecisionTree"]=False
            img_base64 = GaussianNB(processed_data)
            
            os.remove('dataset.csv')  # Remove uploaded file
            # Return data as JSON
            return jsonify({'confusion_matrix': img_base64})
    
    elif route_accessed["upload_RandomForest"] == True:
            from ids_logic import RandomForest
            route_accessed["upload_KNN"]=False
            route_accessed["upload_DecisionTree"]=False
            img_base64 = RandomForest(processed_data)
            
            os.remove('dataset.csv')  # Remove uploaded file
            # Return data as JSON
            return jsonify({'confusion_matrix': img_base64})
    
    else:
        return render_template("model.html")
@app.route('/result')
def result():
    return process()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))