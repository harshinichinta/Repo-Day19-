import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Title
st.title("Random Forest Classification")

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Original Dataset")
    st.write(data.head())

    # Fill missing values instead of dropping
    data = data.fillna(0)

    # Remove duplicates
    data = data.drop_duplicates()

    st.subheader("Dataset Shape")
    st.write(data.shape)

    # Check empty dataset
    if data.shape[0] == 0:
        st.error("Dataset is empty")
        st.stop()

    # Select target column
    target_column = st.selectbox(
        "Select Target Column",
        data.columns
    )

    # Encode categorical columns
    encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

    # Features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Convert all columns to numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Replace NaN values
    X = X.fillna(0)

    # Check minimum rows
    if len(X) < 2:
        st.error("Not enough data for training")
        st.stop()

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Accuracy")
    st.write(accuracy)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax
    )

    st.pyplot(fig)