import numpy as np
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    prediction = model.predict([request.form.values()])

    return render_template('index.html', prediction_text='Illness? {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)