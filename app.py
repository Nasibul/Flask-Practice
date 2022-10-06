from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pickle', 'rb'))
encoder = pickle.load(open('encoder.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inputs=[]
    inputs.append(str(request.form.get("City")))
    inputs.append(str(request.form.get("Gender")))
    inputs.append(int(request.form.get("age")))
    inputs.append(float(request.form.get("income")))
    
    num = inputs[2:]
    cat = inputs[0:2]
    cat = encoder.transform([cat])
    inputs = pd.DataFrame(list(cat.flat) + num).T

    prediction = model.predict(inputs)[0]
    
    return render_template('index.html', prediction_text='Illness: {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)