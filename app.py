import io
import base64
import pandas as pd
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt


from flask import Flask, flash, render_template,request,redirect, url_for, jsonify
from sklearn.tree import DecisionTreeClassifier
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
from intrusion_detection import confusion_matrixDecisionTreeClassifier,confusion_matrixKNN
from intrusion_detection import load_dataset, col_names,confusion_matrix


app = Flask(__name__)
app.config['SECRET_KEY']='supersecret'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.debug = True


route_accessed = {"upload_RandomForest": False, "upload_DecisionTree": False, "upload_SVM": False}


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload FIle")


@app.route("/")
def home():
    return render_template("index.html", model_url = url_for)

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/upload_RandomForest")
def upload_RandomForest():

    route_accessed["upload_RandomForest"] = True

    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_DecisionTree")
def upload_DecisionTree():

    route_accessed["upload_DecisionTree"]=True

    form = UploadFileForm()
    return render_template('upload.html', form=form)

@app.route("/upload_SVM")
def upload_SVM():

    route_accessed["upload_SVM"] = True

    form = UploadFileForm()
    return render_template('upload.html', form=form)


@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']
    file_path = "/" + file.filename
    file.save("dataset.csv")


   
    if request.method == 'POST':

        #check for if file is empty
        if file.filename == '':
            return "Error: No file selected for upload"
        
        #read the uploaded dataset
        try:
            df = pd.read_csv('dataset.csv', header=None, names=col_names)
        except pd.errors.EmptyDataError:
            return "Error: Uploaded file is empty or contains no data"
        
        #check is dataframe is empty
        if df.empty:
            return "Error: Uploaded file is empty"
      
        if route_accessed["upload_DecisionTree"] == True:

            #load the trained model
            with open('IDS_model_DECISION TREE CLASSIFIER.pkl', "rb") as file:
                clf = pickle.load(file)

           
            # Save confusion matrix plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrixDecisionTreeClassifier, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            #plt.tight_layout()
            plt.savefig('confusion_matrix.png')

            # Convert plot to base64 for display in HTML
            with open('confusion_matrix.png', 'rb') as img_file:
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

            os.remove('file')  # Remove uploaded file

            return render_template('result.html', confusion_matrix=img_base64)


        elif route_accessed["upload_RandomForest"] == True:

             with open('intrusion_detection_model.pkl', "rb") as file:
                clf = pickle.load(file)

                 # Save confusion matrix plot
                plt.figure(figsize=(8, 6))
                sns.heatmap(confusion_matrixKNN, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted Label')
                plt.ylabel('True Label')
                #plt.tight_layout()
                plt.savefig('confusion_matrix.png')

                # Convert plot to base64 for display in HTML
                with open('confusion_matrix.png', 'rb') as img_file:
                    img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

                

                return render_template('result.html', confusion_matrix=img_base64)



        else:

            with open('IDS_model_SUPPORT VECTOR MACHINE.pkl', "rb") as file:
                clf = pickle.load(file)


        #load the trained model
        with open('intrusion_detection_model.pkl', 'rb') as file:
            clf = pickle.load(file)

        #Check if 'label' column is present in the DataFrame
     #   if 'label' not in df.columns:
      #    return "Error: 'label' column not found in the uploaded file"

        
         #Perform prediction only on features (exclude 'label' column)
       # features = df.drop(columns=['label'])

        # Perform intrusion detection
       # predictions = clf.predict(features)
    
   
    

    return 'Well done! File uploaded sucessfully'




if __name__ == '__main__':
    app.run(port=6500)