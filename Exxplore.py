import streamlit as st
import pandas as pd
#import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px    
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# for generating Graphs 

st.set_option('deprecation.showPyplotGlobalUse', False)

st.write("""
# Credit Card Fraud Prediction 
This app predicts the **Credit card fraud Detection**!
""")

st.write('---')
st.write('**Description of Dataset**')
# describe all the dataset features using stwrite 

#---------------------------------#
# Sidebar - Collects user input features into dataframe
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    
# Displays the dataset
st.subheader('1. Dataset')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Glimpse of dataset**')
    st.write(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')

