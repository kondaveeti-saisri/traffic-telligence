from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('../model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'holiday': int(request.form['holiday']),
            'temp': float(request.form['temp']),
            'rain': float(request.form['rain']),
            'snow': float(request.form['snow']),
            'weather': int(request.form['weather']),
            'year': int(request.form['year']),
            'month': int(request.form['month']),
            'day': int(request.form['day']),
            'hour': int(request.form['hours']),
            'minute': int(request.form['minutes']),
            'second': int(request.form['seconds'])
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Predict
        prediction = int(model.predict(input_df)[0])
        
        return render_template('index.html', prediction_text=f"Predicted Traffic Volume: {prediction} vehicles/hour")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)