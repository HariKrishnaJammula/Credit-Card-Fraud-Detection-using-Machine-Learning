import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# Title of the app
st.title("Credit Card Fraud Detection Using Machine Learning Algorithms")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the file
    data = pd.read_csv(uploaded_file)

    # Show the first 5 rows of the data
    st.write("First 5 rows of the data:")
    st.write(data.head())

    # Show the shape of the data
    st.write("Shape of the data:")
    st.write(data.shape)

    # Show the data types of the columns
    st.write("Data types of the columns:")
    st.write(data.dtypes)

    # Show the summary statistics of the data
    st.write("Summary statistics of the data:")
    st.write(data.describe())

    # Show the missing values in the data
    st.write("Missing values in the data:")
    st.write(data.isnull().sum())

    # Show the correlation matrix of the data
    st.write("Correlation matrix of the data:")
    st.write(data.corr())


 # Plot scatterplot matrix for all numeric columns
 #   st.write("Scatterplot matrix for all numeric columns:")
 #   sns.pairplot(data.select_dtypes(include=['int64', 'float64']))
 #   st.pyplot()


    fraud_counts = data['fraud'].value_counts()

# Create a pie chart with the fraud counts
    fig, ax = plt.subplots()
    ax.pie(fraud_counts, labels=fraud_counts.index, autopct='%1.1f%%')
    ax.set_title('Fraudulent Transactions')
    ax.axis('equal')
    st.pyplot(fig)


    # Count the number of fraud and non-fraud records
    fraud_count = len(data[data['fraud'] == 1])
    non_fraud_count = len(data[data['fraud'] == 0])

# Calculate the percentages
    fraud_percentage = fraud_count / len(data) * 100
    non_fraud_percentage = non_fraud_count / len(data) * 100

# Create the bar graph
    labels = ['Fraud', 'Non-Fraud']
    percentages = [fraud_percentage, non_fraud_percentage]

# Display the chart using Streamlit
    st.title('Percentage of Fraud and Non-Fraud Records')
    st.bar_chart({'Labels': labels, 'Percentages': percentages})
    

# Create your heatmap
    st.title("Correlation Matrix (Heat Map)")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr().round(3), annot=True, vmin=-1, vmax=1, cmap="coolwarm", ax=ax)
    sns.set(rc={"figure.figsize":(10,10)})

# Display the plot in Streamlit app
    st.pyplot(fig)

    st.title('Fraud Data EDA')

# Set the figure size
    plt.figure(figsize=(10, 10))

# Create a subplot for each variable of interest
    plt.subplot(2, 2, 1)
    data['repeat_retailer'].value_counts().plot(kind='bar', color='purple')
    plt.title('Repeat Retailer')

    plt.subplot(2, 2, 2)
    data['used_chip'].value_counts().plot(kind='bar', color='green')
    plt.title('Used Chip')

    plt.subplot(2, 2, 3)
    data['used_pin_number'].value_counts().plot(kind='bar', color='blue')
    plt.title('Used Pin Number')

    plt.subplot(2, 2, 4)
    data['online_order'].value_counts().plot(kind='bar', color='red')
    plt.title('Online Order')

# Set the overall title of the figure
    plt.suptitle('Fraud Data EDA')

# Display the figure using Streamlit
    st.pyplot(plt)

    def plot_categorical_vars(data, cat_vars):
    # set plot style
        sns.set_style("whitegrid")

    # set up figure and axis objects
        fig, axs = plt.subplots(len(cat_vars), 2, figsize=(10, 4 * len(cat_vars)))
    
    # set up colors and explode values for the pie charts
        colors = ["#66b3ff", "#ffcc99"]
        explode = (0, 0.1)

    # create pie chart
        pie_data = data[cat_vars].apply(pd.Series.value_counts)
        for i, var in enumerate(cat_vars):
            axs[i, 0].pie(pie_data[var], labels=pie_data[var].index, autopct="%.2f%%", colors=colors, explode=explode)
            axs[i, 0].set_title(f"Distribution of {var}", fontsize=10)

        # create bar plot
            sns.countplot(x=var, data=data, ax=axs[i, 1])
            axs[i, 1].set_title(f"Count of {var} by Class", fontsize=10)
            axs[i, 1].set_xlabel(var, fontsize=16)
            axs[i, 1].set_ylabel("Count", fontsize=16)

        plt.tight_layout()
        st.pyplot(fig)

    cat_vars = ["repeat_retailer", "used_chip", "used_pin_number", "online_order"]

    st.title("Categorical Variables Plot")
    plot_categorical_vars(data, cat_vars)



    # Define the feature columns and target variable
    feature_columns = ["distance_from_home", "distance_from_last_transaction",
                       "ratio_to_median_purchase_price", "repeat_retailer",
                       "used_chip", "used_pin_number", "online_order"]
    target_variable = "fraud"

    # Split the data into training and testing sets
    X = data[feature_columns]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

    # Display the dataset
    st.subheader("Dataset")
    st.write(data)

    # Display the feature columns and target variable
    st.subheader("Feature Columns")
    st.write(feature_columns)
    st.subheader("Target Variable")
    st.write(target_variable)

    # Display the training and testing sets
    st.subheader("Training and Testing Sets")
    st.write("X_train:", X_train.shape)
    st.write("y_train:", y_train.shape)
    st.write("X_test:", X_test.shape)
    st.write("y_test:", y_test.shape)




    # Create a logistic regression model and fit it to the training data
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = logreg.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Display the accuracy score
    st.subheader("Accuracy of logistic regression classifier on test set:")
    st.write("{:.5f}".format(accuracy))

    # Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test, y_pred, digits=6)
    st.code(report, language="text")



    # Create a decision tree model and fit it to the training data
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = decision_tree.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test, y_pred)

    # Display the accuracy score
    st.subheader("Accuracy of decision tree classifier on test set:")
    st.write("{:.5f}".format(accuracy))

    # Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test, y_pred, digits=6)
    st.code(report, language="text")


# Create a Random Forest Classifier and fit it to the training data
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train, y_train)

# Make predictions on the test data
    y_pred = rfc.predict(X_test)

# Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test, y_pred)

# Display the accuracy score
    st.subheader("Accuracy of Random Forest Classifier on test set:")
    st.write("{:.5f}".format(accuracy))

# Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test, y_pred, digits=6)
    st.code(report, language="text")
