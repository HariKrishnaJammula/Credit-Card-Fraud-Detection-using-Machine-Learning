import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title('Credit Card Fraud Detection')
st.text('Web app for my credit card fraud up')

uploaded_file = st.file_uploader('Upload your file')


if uploaded_file:
    st.header('Data Statistics')
    df = pd.read_csv(uploaded_file)
    st.write(df.describe())

    st.header('Data header')
    st.write(df.head())

    fraud_count = len(df[df['fraud'] == 1])
    non_fraud_count = len(df[df['fraud'] == 0])

# Calculate the percentages
    fraud_percentage = fraud_count / len(df) * 100
    non_fraud_percentage = non_fraud_count / len(df) * 100

# Create the bar graph
    labels = ['Fraud', 'Non-Fraud']
    percentages = [fraud_percentage, non_fraud_percentage]

    plt.bar(labels, percentages)
    plt.title('Percentage of Fraud and Non-Fraud Records')
    plt.xlabel('Type of Record')
    plt.ylabel('Percentage')
    plt.show()

