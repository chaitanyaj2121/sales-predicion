from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os

# Initialize the Flask app
app = Flask(__name__)

# Load and prepare the data
df = pd.read_csv("advertising.csv")
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to get predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Accept JSON data from frontend
    tv_ad_spend = float(data['tv_ad_spend'])  # Get 'tv_ad_spend' from JSON
    radio_ad_spend = float(data['radio_ad_spend'])  # Get 'radio_ad_spend' from JSON
    newspaper_ad_spend = float(data['newspaper_ad_spend'])  # Get 'newspaper_ad_spend' from JSON

    # Prepare input for prediction
    input_data = np.array([[tv_ad_spend, radio_ad_spend, newspaper_ad_spend]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Return the result as JSON
    return jsonify({
        'predicted_sales': prediction
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
