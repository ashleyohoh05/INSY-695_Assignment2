import streamlit as st
import joblib
import numpy as np

# Load the trained pipeline
try:
    pipeline = joblib.load("model.pkl")
    model = pipeline["model"]
    scaler = pipeline["scaler"]
    imputer = pipeline["imputer"]
    st.write("Model loaded successfully.")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title and description
st.title("Tumor Area Prediction")
st.write("Predict tumor area based on input features using a Gradient Boosting Regressor.")

# Input fields for user input
feature1 = st.number_input("Enter Feature 1 (e.g., radius1):", step=0.01, format="%.2f")
feature2 = st.number_input("Enter Feature 2 (e.g., texture1):", step=0.01, format="%.2f")
feature3 = st.number_input("Enter Feature 3 (e.g., perimeter1):", step=0.01, format="%.2f")

# Early validation of inputs
if feature1 < 0 or feature2 < 0 or feature3 < 0:
    st.error("Feature values must be non-negative.")
elif feature1 > 100 or feature2 > 100 or feature3 > 100:
    st.error("Feature values must be realistic and below 100 (example).")
else:
    # Prediction button
    if st.button("Predict"):
        try:
            # Additional validation inside the prediction block
            input_data = np.array([[feature1, feature2, feature3]])
            if np.isnan(input_data).any():
                st.error("Please fill in all the fields with valid values.")
            else:
                # Preprocess inputs
                input_data_imputed = imputer.transform(input_data)
                input_data_scaled = scaler.transform(input_data_imputed)

                # Predict
                prediction = model.predict(input_data_scaled)
                st.success(f"Predicted Tumor Area: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")
