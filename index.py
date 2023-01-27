from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import tensorflow as tf


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    quantity = data['quantity']
    volume = data['volume']
    distance = data['distance']
    
    input_data = np.array([quantity, volume, distance]).reshape(1, -1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(input_data)
    
    model = tf.keras.models.load_model('model.h5', compile=False)

    prediction = model.predict(scaled_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run()(debug=False,host='0.0.0.0')
