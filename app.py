import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    from imblearn.over_sampling import SMOTE
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
# Define the features and target variables
    x = data.drop("fraud", axis=1).values
    y = data["fraud"].values

# Apply SMOTE to balance the data
    smote = SMOTE(random_state=39)
    non_fraud_over, fraud_over = smote.fit_resample(x, y)

# Create a new DataFrame for the balanced data
    non_fraud_over_df = pd.DataFrame(non_fraud_over, columns=["distance_from_home", "distance_from_last_transaction",
        "ratio_to_median_purchase_price", "repeat_retailer", "used_chip",
        "used_pin_number", "online_order"])
    non_fraud_over_df["fraud"] = fraud_over
    osdf = non_fraud_over_df

# Display the shape and info of the new DataFrame
    st.write("osdf shape:", osdf.shape)
    st.write(osdf.info())

# Display the descriptive statistics of the new DataFrame
    st.write(osdf.describe())

# Display the correlation matrix as a heatmap
    sns.set(rc={"figure.figsize":(10,10)})
    sns.heatmap(osdf.corr().round(3), annot=True, vmin=-1, vmax=1, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    st.pyplot()

# Split the data into training and testing sets
    feature_columns = ["distance_from_home", "distance_from_last_transaction",
        "ratio_to_median_purchase_price", "repeat_retailer", "used_chip", "used_pin_number", "online_order"]
    X_smote = osdf[feature_columns]
    y_smote = osdf.fraud
    X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.2, random_state=39)

# Display the shapes of the training and testing sets
    st.write("X_train_smote shape:", X_train_smote.shape)
    st.write("X_test_smote shape:", X_test_smote.shape)
    st.write("y_train_smote shape:", y_train_smote.shape)
    st.write("y_test_smote shape:", y_test_smote.shape)



    import streamlit as st
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    # Create a logistic regression model and fit it to the training data
    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train_smote, y_train_smote)

    # Make predictions on the test data
    y_pred = logreg.predict(X_test_smote)

    # Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test_smote, y_pred)

    # Display the accuracy score
    st.subheader("Accuracy of logistic regression classifier on test set:")
    st.write("{:.5f}".format(accuracy))

    # Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test_smote, y_pred, digits=6)
    st.code(report, language="text")

    import streamlit as st
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import metrics

    # Create a decision tree model and fit it to the training data
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train_smote, y_train_smote)

    # Make predictions on the test data
    y_pred = decision_tree.predict(X_test_smote)

    # Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test_smote, y_pred)

    # Display the accuracy score
    st.subheader("Accuracy of decision tree classifier on test set:")
    st.write("{:.5f}".format(accuracy))

    # Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test_smote, y_pred, digits=6)
    st.code(report, language="text")


    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics
    from sklearn.metrics import classification_report


    import streamlit as st
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import metrics

# Create a Random Forest Classifier and fit it to the training data
    rfc = RandomForestClassifier(n_estimators=100)
    rfc.fit(X_train_smote, y_train_smote)

# Make predictions on the test data
    y_pred = rfc.predict(X_test_smote)

# Calculate the accuracy of the model
    accuracy = metrics.accuracy_score(y_test_smote, y_pred)

# Display the accuracy score
    st.subheader("Accuracy of Random Forest Classifier on test set:")
    st.write("{:.5f}".format(accuracy))

# Display the classification report
    st.subheader("Classification Report")
    report = metrics.classification_report(y_test_smote, y_pred, digits=6)
    st.code(report, language="text")
