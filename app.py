from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    feature = [int(x) for x in request.form.values()]
    feature = [feature]
    feature = MinMaxScaler().fit_transform(feature)
    result = model.predict(feature)
    return render_template('index.html', prediction_text=f'The predicted value of insurance price is ${int(result[0])}')


if __name__ == '__main__':
    app.run(debug=True)
