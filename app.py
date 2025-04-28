import streamlit as st
import pandas as pd
import pickle

# ------------------------
# Load the saved pkl models
with open('mkmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Cluster colors
cluster_colors = {
    0: '#FF6961',
    1: '#77DD77',
    2: '#84B6F4',
    3: '#FFD700',
}

# ------------------------
# Streamlit UI
st.set_page_config(page_title="Customer Segmentation App", layout="centered")

st.title("🛍️ Customer Segmentation Prediction App")

st.write("Fill the information below to predict the customer segment:")

# User Inputs
age = st.slider('Age', 18, 70, 30)
gender = st.selectbox('Gender', ['Male', 'Female'])
income = st.number_input('Annual Income ($)', min_value=10000, max_value=100000, value=50000)
spending_score = st.slider('Spending Score (1-100)', 1, 100, 50)

# Predict Button
if st.button('Predict Segment'):
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [0 if gender == 'Male' else 1],
        'Annual Income': [income],
        'Spending Score': [spending_score]
    })

    input_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(input_scaled)[0]

    st.markdown(f"<h2 style='color:{cluster_colors[cluster]}'>Predicted Cluster: {cluster}</h2>", unsafe_allow_html=True)

    st.subheader("Cluster Color Guide")
    cluster_table = pd.DataFrame({
        'Cluster': [0, 1, 2, 3],
        'Color': ['Red', 'Green', 'Blue', 'Yellow'],
        'Meaning': ['Low Income, Low Spending', 'High Income, High Spending', 'Medium Income, Medium Spending', 'Special Category']
    })

    def color_row(row):
        color = cluster_colors[row['Cluster']]
        return [f'background-color: {color}'] * len(row)

    st.dataframe(cluster_table.style.apply(color_row, axis=1))
