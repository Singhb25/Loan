from flask import Flask, render_template, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join('model', 'ML_Model1.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        features = [float(x) for x in request.form.values()]
        
        # Assuming model expects a list of features
        prediction = model.predict([features])
        
        return jsonify({
            'prediction': int(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
