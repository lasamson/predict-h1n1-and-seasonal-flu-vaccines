import pickle

from flask import Flask
from flask import request
from flask import jsonify

# Saved model file
model_file = 'bin/final_model.bin'

# Load trained DictVectorizers and XGBoost classifier models for h1n1 and seasonal
# flu prediction tasks
with open(model_file, 'rb') as f_in:
    (dv_h1n1, xgbc_cv_h1n1, dv_seasonal, xgbc_cv_seasonal) = pickle.load(f_in)

# Create Flask app
app = Flask('vaccine_prediction')

@app.route('/predict', methods=['POST'])
def predict():
    # Get respondent
    respondent = request.get_json()

    # Featurize for h1n1 and seasonal flu prediction
    X_h1n1 = dv_h1n1.transform([respondent])
    X_seasonal = dv_seasonal.transform([respondent])

    # Get h1n1 probability and make decision
    y_pred_h1n1 = xgbc_cv_h1n1.predict_proba(X_h1n1)[0, 1]
    h1n1 = y_pred_h1n1 >= 0.5

    # Get seasonal flu probability and make decision
    y_pred_seasonal = xgbc_cv_seasonal.predict_proba(X_seasonal)[0, 1]
    seasonal = y_pred_seasonal >= 0.5

    # Create result object to return
    result = {
        'h1n1_probability': float(y_pred_h1n1),
        'h1n1': float(h1n1),
        'seasonal_probabilty': float(y_pred_seasonal),
        'seasonal': float(seasonal)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

    