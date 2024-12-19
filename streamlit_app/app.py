import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# Load preprocessed data
@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('../data/processed/preprocessed_data.csv') 
    return data

# Load the Random Forest model
@st.cache_resource
def load_model():
    # Load the best model pipeline (preprocessing + Random Forest)
    model = joblib.load('../models/best_model.pkl') 
    return model

# Load test data for evaluation
@st.cache_data
def load_train_test_data():
    # Load train-test split data
    X_test = pd.read_csv('../data/processed/X_test.csv')  
    y_test = pd.read_csv('../data/processed/y_test.csv')
    return X_test, y_test

# Streamlit app sections
def main():
    st.title("Beijing Air Quality Analysis and Prediction")
    st.sidebar.title("Navigation")
    menu = ["Data Overview", "Exploratory Data Analysis", "Model Evaluation", "Prediction"]
    choice = st.sidebar.radio("Go to", menu)

    # Load data and model
    df = load_data()
    model = load_model()
    X_test, y_test = load_train_test_data()

    # Section 1: Data Overview
    if choice == "Data Overview":
        st.header("Data Overview")
        st.write("### Dataset Information")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.write("### Sample Data")
        st.write(df.head())
        st.write("### Data Description")
        st.write(df.describe())

    # Section 2: Exploratory Data Analysis (EDA)
    elif choice == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")

        # Univariate Analysis: Numerical Columns
        st.subheader("Univariate Analysis: Distributions of Numerical Columns")
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        for col in numerical_columns:
            plt.figure(figsize=(10, 5))
            sns.histplot(df[col], kde=True, bins=30, color='skyblue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            st.pyplot(plt)

        # Univariate Analysis: Categorical Columns
        st.subheader("Univariate Analysis: Categorical Columns")
        categorical_columns = df.select_dtypes(include=["object"]).columns
        for col in categorical_columns:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind="bar", color="skyblue", edgecolor="black")
            plt.title(f"Frequency of Categories in {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            st.pyplot(plt)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["float64", "int64"])
        if not numeric_df.empty:
            plt.figure(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(plt)
        else:
            st.warning("No numeric columns available for correlation heatmap.")

        # Scatter Plots for Highly Correlated Variables
        st.subheader("Scatter Plots for Highly Correlated Variables")
        correlation_pairs = [
            ("PM2.5", "PM10"),
            ("TEMP", "DEWP"),
            ("CO", "PM2.5"),
            ("TEMP", "PM2.5"),
        ]
        for x, y in correlation_pairs:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x=x, y=y, alpha=0.7)
            plt.title(f"Scatter Plot: {x} vs {y}")
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(plt)

        # Pairplot
        st.subheader("Pairplot of Selected Variables")
        selected_columns = ["PM2.5", "PM10", "TEMP", "DEWP", "CO"]
        sns.pairplot(df[selected_columns], diag_kind="kde")
        st.pyplot(plt)

    # Section 3: Model Evaluation
    elif choice == "Model Evaluation":
        
        st.header("Model Evaluation")

        # Predictions on test data
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display metrics
        st.write(f"### Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"### R-Squared (R²): {r2:.2f}")

        # Residual Plot
        st.subheader("Residual Plot")
        residuals = y_test.values.flatten() - y_pred
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30, color='blue')
        plt.title("Residuals Distribution")
        plt.xlabel("Residuals")
        plt.ylabel("Frequency")
        st.pyplot(plt)

        # Bar plots for Model Performance Comparison
        st.subheader("Model Performance Comparison")
        results_df = pd.read_csv("../models/model_comparison.csv")  
        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x="R²", y="Model", palette="viridis")
        plt.title("Model Performance Comparison (R² Scores)")
        st.pyplot(plt)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=results_df, x="MSE", y="Model", palette="viridis")
        plt.title("Model Performance Comparison (MSE Scores)")
        st.pyplot(plt)

    # Section 4: Prediction
    elif choice == "Prediction":
        st.header("PM2.5 Prediction")
        st.write("Provide input values for the following features:")

        # Extract feature columns (exclude target column)
        feature_columns = df.drop(columns="PM2.5").columns
        input_data = {}

        # Collect user inputs
        for col in feature_columns:
            if df[col].dtype in ["float64", "int64"]:
                input_data[col] = st.number_input(f"{col} (numeric)", value=float(df[col].mean()))
            else:
                input_data[col] = st.selectbox(f"{col} (categorical)", options=df[col].unique())

        # Predict button
        if st.button("Predict"):
            # Convert user inputs to a DataFrame
            input_df = pd.DataFrame([input_data])

            # Make predictions using the loaded model
            prediction = model.predict(input_df)
            st.success(f"Predicted PM2.5 level: {prediction[0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()