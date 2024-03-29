from flask import Flask, flash, render_template,request,redirect, url_for, jsonify
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import io
import base64
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from intrusion_detection import train_model
from intrusion_detection import load_dataset, col_names, train_model
import pickle

app = Flask(__name__)
app.config['SECRET_KEY']='supersecret'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.debug = True


class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload FIle")


@app.route("/")
def home():
    return render_template("index.html", model_url = url_for)

@app.route("/model")
def model():
    return render_template("model.html")

@app.route("/upload")
def upload():
    form = UploadFileForm()
    return render_template('upload.html', form=form)


@app.route('/submit', methods=['POST'])
def submit():
    file = request.files['file']
    file_path = "/" + file.filename
    file.save("dataset.csv")
   
    if request.method == 'POST':
      
        df = pd.read_csv('dataset.csv', header=None, names = col_names)
        

        with open('intrusion_detection_model.pkl', 'rb') as file:
            clf = pickle.load(file)
        predictions = clf.predict(df)
        attack_count = sum(predictions)
        # Redirect to result page or do whatever further processing is needed
        return redirect(url_for('result.html'))
  
   
    

    return 'Well done! File uploaded sucessfully'




if __name__ == '__main__':
    app.run()