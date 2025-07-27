import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px

# --- 2. Streamlit App Layout Configuration (MUST BE FIRST) ---
# This line must be the very first Streamlit command in your script.
st.set_page_config(page_title="ML Model Deployment App", layout="centered")

# --- 1. Load Data and Train Model (Re-creating from notebook) ---
# In a real scenario, you would load a pre-trained model (e.g., using joblib or pickle)
# and a pre-fitted scaler. For this example, we'll re-train for simplicity
# since we don't have the saved model files.

@st.cache_resource # Cache the model and scaler to avoid re-training on each rerun
def load_and_train_model():
    """
    Loads the dataset, splits it, scales features, and trains the best performing model
    (Logistic Regression with tuned hyperparameters) and returns the scaler.
    """
    try:
        # Assuming 'classification_dataset.csv' is available in the same directory
        df = pd.read_csv("classification_dataset.csv")
    except FileNotFoundError:
        st.error("classification_dataset.csv not found. Please ensure it's in the same directory.")
        st.stop() # Stop the app if the file is not found

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test) # Not needed for training, but kept for context

    # Re-training the best model found in the notebook: Logistic Regression (Tuned)
    # Best parameters from the notebook: C=10, solver='liblinear'
    model = LogisticRegression(C=10, solver='liblinear', max_iter=1000)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns.tolist() # Return feature names for input fields

model, scaler, feature_names = load_and_train_model()

st.title("Breast Cancer Prediction App")
st.markdown("Enter the patient's measurements to get a prediction on whether the tumor is Benign (0) or Malignant (1).")

st.sidebar.header("Input Features")

# Create input fields for all 30 features
input_data = {}
for feature in feature_names:
    # Using number_input for numerical features. Adjust min_value and max_value
    # based on typical ranges of your dataset features if known.
    # For a more robust app, you'd analyze min/max from training data.
    # For now, using a generic range or allowing user to input freely.
    # Let's try to infer a reasonable default based on the first few rows of the dataset
    # from the notebook snippet.
    # Since we don't have the full dataset here, we'll use a generic approach.
    # A more advanced solution would involve loading a small sample to get min/max.

    # Placeholder for min/max values - ideally derived from X_train
    # For demonstration, we'll use generic values or slightly more informed ones
    # based on the head of the dataset from the notebook.
    if "radius" in feature:
        min_val, max_val = 5.0, 30.0
    elif "texture" in feature:
        min_val, max_val = 5.0, 40.0
    elif "perimeter" in feature:
        min_val, max_val = 40.0, 200.0
    elif "area" in feature:
        min_val, max_val = 100.0, 2500.0
    elif "smoothness" in feature:
        min_val, max_val = 0.05, 0.2
    elif "compactness" in feature:
        min_val, max_val = 0.0, 0.4
    elif "concavity" in feature:
        min_val, max_val = 0.0, 0.5
    elif "concave points" in feature:
        min_val, max_val = 0.0, 0.2
    elif "symmetry" in feature:
        min_val, max_val = 0.1, 0.4
    elif "fractal dimension" in feature:
        min_val, max_val = 0.05, 0.1
    else: # For worst and standard error features, broader range
        min_val, max_val = 0.0, 5000.0 # Area can be large, others smaller

    # Get a default value for the input field. Using a placeholder for now.
    # In a real app, you might use the mean of the training data.
    default_val = (min_val + max_val) / 2 # Simple average for default

    input_data[feature] = st.sidebar.number_input(
        f"Enter {feature}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=0.001,
        format="%.4f" # Format to 4 decimal places for better precision
    )

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

st.subheader("Input Data Overview")
st.write(input_df)

# --- 3. Prediction ---
if st.button("Get Prediction"):
    try:
        # Scale the input data using the trained scaler
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        st.subheader("Prediction Result")
        if prediction[0] == 0:
            st.success("The model predicts: Benign (0)")
        else:
            st.error("The model predicts: Malignant (1)")

        st.write(f"Confidence (Probability of Benign): {prediction_proba[0][0]:.4f}")
        st.write(f"Confidence (Probability of Malignant): {prediction_proba[0][1]:.4f}")

        # --- 4. Visualization of Model Output ---
        st.subheader("Prediction Probability Visualization")
        proba_df = pd.DataFrame({
            'Class': ['Benign', 'Malignant'],
            'Probability': [prediction_proba[0][0], prediction_proba[0][1]]
        })

        fig = px.bar(
            proba_df,
            x='Class',
            y='Probability',
            color='Class',
            title='Prediction Probabilities',
            labels={'Probability': 'Probability', 'Class': 'Tumor Type'},
            color_discrete_map={'Benign': 'lightgreen', 'Malignant': 'salmon'}
        )
        fig.update_layout(xaxis_title="Tumor Type", yaxis_title="Probability", yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure all input fields have valid numerical values.")

st.markdown("""
---
### About This Application

This application demonstrates the deployment of a machine learning model for breast cancer prediction.

**Dataset:**
The model is trained on a dataset containing various characteristics of cell nuclei, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. These features are derived from digitized images of fine needle aspirates (FNA) of breast masses.

**Model Used:**
The core of this application is a **Logistic Regression** model. As a hyperparameter-tuned Logistic Regression model was found to be the best performer in terms of F1 Score. The model processes the input features (which are scaled to a standard range) to predict the likelihood of a tumor being benign or malignant.

**Interpreting Results:**
* **Benign (0):** Indicates the tumor is non-cancerous.
* **Malignant (1):** Indicates the tumor is cancerous.
The "Confidence (Probability)" values show the model's certainty for each class. A higher probability for "Malignant" suggests a greater likelihood of a cancerous tumor, and vice-versa for "Benign."

**Important Note:**
This application is for **demonstration and educational purposes only** and should not be used for actual medical diagnosis. Always consult with a qualified healthcare professional for any health concerns.
""")
