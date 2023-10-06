import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('d.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    '''features = [[int(x) for x in request.form.values()]]
    pred = model.predict(features)''' 
    
    if request.method == 'POST':
      i1 = request.form['No.of Bedrooms']
      i2 = request.form['Area']
      i3 = request.form['Parking']
      i4 = request.form['Power_backup']
      i5 = request.form['Resale']
    inputs=[[i1,i2,i3,i4,i5]]
    pred = model.predict(inputs)
    result=round(pred[0])

    return render_template('d.html', prediction='House Price Is Rs : {}'.format(result))



if __name__ == "__main__":
    app.run(debug=True)
    