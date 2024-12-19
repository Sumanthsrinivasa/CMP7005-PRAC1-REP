# CMP7005-PRAC1: Beijing Air Quality Analysis and Prediction

This project analyzes air quality data in Beijing, implements machine learning models to predict PM2.5 levels, and visualizes insights using a Streamlit app. The repository demonstrates exploratory data analysis (EDA), preprocessing, model building, and deployment.

---

## Project Structure

```plaintext
CMP7005-PRAC1/
├── data/
│   ├── processed/
│   │   ├── merged_data.csv          # Merged dataset
│   │   ├── preprocessed_data.csv    # Preprocessed dataset
│   │   ├── X_test.csv               # Test features
│   │   ├── y_test.csv               # Test target values
│   └── raw/                         # Raw datasets
├── models/
│   ├── best_model.pkl               # Saved Random Forest pipeline
│   ├── model_comparison.csv         # Model performance comparison
├── streamlit_app/
│   ├── app.py                       # Streamlit application for EDA and prediction
├── CMP7005_PRAC1.ipynb              # Jupyter notebook with the full 
├── requirements.txt                 # Dependencies for the project
├── README.md                        # Project documentation
```

## How to Run the Project

1. Clone the Repository

    Clone the repository to your local machine:

        git clone <repository_url>
        cd CMP7005-PRAC1

2. Set Up the Environment

Create a virtual environment and install the required packages:
    ```bash

    #Create a virtual environment (optional but recommended)
    python -m venv myenv

    # Activate the virtual environment
    # On Windows:
    myenv\Scripts\activate
    # On macOS/Linux:
    source myenv/bin/activate

    # Install dependencies
        pip install -r requirements.txt
    ```

3. Run the Jupyter Notebook

    Open and execute the notebook CMP7005_PRAC1.ipynb to:
    -	Merge and preprocess the data.
    -	Perform EDA.
    -	Train, evaluate, and save the machine learning model.

    To run the notebook:

        jupyter notebook CMP7005_PRAC1.ipynb

4. Run the Streamlit App

    Start the Streamlit app for interactive visualization and predictions:

        streamlit run streamlit_app/app.py

    Open the app in your browser at http://localhost:8501.

## Features

1. Data Processing

    -	Merged raw datasets into a single file.
    -	Preprocessed data by handling missing values, encoding categorical variables, and scaling numerical variables.

2. Exploratory Data Analysis (EDA)
	-	Visualized numerical data distributions.
	-	Generated correlation heatmaps and scatterplots.
	-	Explored categorical variables using bar charts and boxplots.

3. Machine Learning
	-	Trained multiple regression models, including:
        -	Random Forest
        -	Linear Regression
        -	Gradient Boosting
        -	Decision Tree
        -	K-Nearest Neighbors
	-	Evaluated models using metrics like Mean Squared Error (MSE) and R².
	-	Saved the best-performing model (Random Forest) as a pipeline.

4. Streamlit App
	-	Interactive EDA visualizations, including histograms, scatterplots, and heatmaps.
	-	Predict PM2.5 levels based on user-provided input features.

## Version Control

All project progress is version-controlled using Git. Below is a snippet of the commit history:


## Technologies Used
-	Python: Core programming language
-	Libraries:
    -	pandas, numpy: Data manipulation
    -	matplotlib, seaborn: Visualization
    -	scikit-learn: Machine learning
-	Streamlit: Web-based visualization and deployment
-	Jupyter Notebook: Development environment
-	Git & GitHub: Version control and repository hosting
