import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Page Title
st.title("AdaBoost Regression on Asteroid Dataset")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    data = pd.read_csv(uploaded_file)

    # Display Dataset
    st.subheader("Dataset")
    st.write(data.head())

    # Dataset Shape
    st.subheader("Dataset Shape")
    st.write(data.shape)

    # Fill Missing Values
    data = data.fillna(0)

    # Remove Duplicates
    data = data.drop_duplicates()

    # Display Columns
    st.subheader("Columns")
    st.write(list(data.columns))

    # Select Target Column
    target_column = st.selectbox(
        "Select Target Column",
        data.columns
    )

    # Encode Categorical Columns
    encoder = LabelEncoder()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = encoder.fit_transform(
                data[col].astype(str)
            )

    # Features and Target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Convert Features to Numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Convert Target to Numeric
    y = pd.to_numeric(y, errors='coerce')
    y = y.fillna(0)

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Base Estimator
    base_model = DecisionTreeRegressor(
        max_depth=4
    )

    # AdaBoost Regressor
    model = AdaBoostRegressor(
        estimator=base_model,
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display Metrics
    st.subheader("Model Evaluation")

    st.write(f"Mean Absolute Error : {mae}")
    st.write(f"Mean Squared Error : {mse}")
    st.write(f"Root Mean Squared Error : {rmse}")
    st.write(f"R2 Score : {r2}")

    # Actual vs Predicted Plot
    st.subheader("Actual vs Predicted")

    fig1, ax1 = plt.subplots(figsize=(8,5))

    ax1.scatter(y_test, y_pred)

    ax1.set_xlabel("Actual Values")
    ax1.set_ylabel("Predicted Values")
    ax1.set_title("Actual vs Predicted")

    st.pyplot(fig1)

    # Error Distribution Plot
    st.subheader("Error Distribution")

    errors = y_test - y_pred

    fig2, ax2 = plt.subplots(figsize=(8,5))

    sns.histplot(
        errors,
        bins=30,
        kde=True,
        ax=ax2
    )

    ax2.set_title("Error Distribution")

    st.pyplot(fig2)

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

    # Feature Importance Plot
    fig3, ax3 = plt.subplots(figsize=(10,6))

    sns.barplot(
        x="Importance",
        y="Feature",
        data=feature_df,
        ax=ax3
    )

    ax3.set_title("Feature Importance")

    st.pyplot(fig3)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    fig4, ax4 = plt.subplots(figsize=(12,8))

    sns.heatmap(
        data.corr(),
        cmap="coolwarm",
        ax=ax4
    )

    ax4.set_title("Correlation Heatmap")

    st.pyplot(fig4)