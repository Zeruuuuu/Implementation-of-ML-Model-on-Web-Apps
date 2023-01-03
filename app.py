from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['PUT'])
def predict():
    feature = [int(x) for x in request.form.values()]
    feature = [feature]
    feature = MinMaxScaler().fit_transform(feature)
    result = model.predict(feature)
    result = int(result[0])
    return render_template('index.html', prediction_text='The predicted value of insurance price is $ {}'.format(result))


if __name__ == '__main__':
    app.run(debug=True)
