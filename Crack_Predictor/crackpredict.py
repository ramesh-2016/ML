import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
import os


# Load the trained model
@st.cache_resource
def load_model():
    with open('Crackprediction_Model2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Predict and save the results
def predict_and_save(input_df):
    model = load_model()
    
    # Check if the required column is in the uploaded data
    if 'IRI' not in input_df.columns:
        st.error("The uploaded CSV file does not contain the required column 'Var1'. Please check your file and try again.")
        return None
    
    # Ensure the column order matches what the model expects
    X = input_df[['IRI']]
    
    # Make predictions
    input_df['CRACK_EXTENT'] = model.predict(X)
    
    return input_df

def main():
    st.title("CSV File Predictor")

    # File upload widget
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded CSV file
        input_df = pd.read_csv(uploaded_file)

        # Display uploaded data
        st.write("Uploaded Data:")
        st.write(input_df)

        # Predict and save results
        result_df = predict_and_save(input_df)

        if result_df is not None:
            # Display results
            st.write("Data with Predictions:")
            st.write(result_df)

            # Provide download link for the result
            result_csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predicted Data as CSV",
                data=result_csv,
                file_name='predicted_data.csv',
                mime='text/csv',
            )

if __name__ == "__main__":
    main()