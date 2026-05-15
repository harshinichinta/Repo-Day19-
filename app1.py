import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Title
st.title("Random Forest Classification")

# Upload CSV File
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read CSV
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset")
    st.write(data.head())

    # Remove missing values
    data = data.dropna()

    # Remove duplicates
    data = data.drop_duplicates()

    st.subheader("Columns")
    st.write(data.columns)

    # Select target column
    target_column = st.selectbox(
        "Select Target Column",
        data.columns
    )

    # Encode ALL categorical columns
    le = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = le.fit_transform(data[col].astype(str))

    # Features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Convert target if categorical
    if y.dtype == 'object':
        y = le.fit_transform(y.astype(str))

    # Train test split
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

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("Accuracy")
    st.write(accuracy)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

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

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)

    # Feature Importance
    importance = model.feature_importances_

    feature_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    })

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=False
    )

    st.subheader("Feature Importance")
    st.write(feature_df)

    # Plot Feature Importance
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_df,
        ax=ax2
    )

    st.pyplot(fig2)