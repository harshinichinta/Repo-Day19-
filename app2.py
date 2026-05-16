import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------
# Page Title
# ------------------------------------------

st.title("AdaBoost Classification")
st.subheader("Online News Popularity Prediction")

# ------------------------------------------
# Load Dataset
# ------------------------------------------

df = pd.read_csv("OnlineNewsPopularity.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# ------------------------------------------
# Display Dataset
# ------------------------------------------

st.write("Dataset Preview")
st.dataframe(df.head())

# ------------------------------------------
# Create Target Variable
# ------------------------------------------

median_shares = df['shares'].median()

df['target'] = np.where(df['shares'] > median_shares, 1, 0)

# ------------------------------------------
# Features and Target
# ------------------------------------------

X = df.drop(['url', 'shares', 'target'], axis=1)
y = df['target']

# ------------------------------------------
# Train Test Split
# ------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------------------
# AdaBoost Model
# ------------------------------------------

base_model = DecisionTreeClassifier(max_depth=1)

# If estimator gives error,
# replace estimator with base_estimator

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ------------------------------------------
# Accuracy
# ------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.write(f"Accuracy: {accuracy:.4f}")

# ------------------------------------------
# Classification Report
# ------------------------------------------

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ------------------------------------------
# Confusion Matrix
# ------------------------------------------

st.subheader("Confusion Matrix Heatmap")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    ax=ax
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# ------------------------------------------
# Feature Importance
# ------------------------------------------

st.subheader("Top 10 Important Features")

importance = model.feature_importances_

feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

feature_df = feature_df.sort_values(
    by='Importance',
    ascending=False
)

top_features = feature_df.head(10)

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=top_features,
    ax=ax2
)

ax2.set_title("Top 10 Important Features")

st.pyplot(fig2)