import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston


##  Random Forest Testing Mode 1::::


# Add Streamlit UI components
#    st.header("Random Forest Classifier")

# Check if the data is loaded and loaded properly
#    clf = RandomForestClassifier(n_estimators=100)
#    clf.fit(X_train,y_train)
#    y_pred_rf = clf.predict(X_test)
    
    # Display the accuracy score
#    st.write("Accuracy:", metrics.accuracy_score(y_test, y_pred_rf))
    
    # Display the classification report
#    st.write("Classification Report:")
#    st.write(classification_report(y_test, y_pred_rf, digits=6))
