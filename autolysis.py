# /// script
#dependencies = [
#    "os",
 #   "pandas",
  #  "numpy",
   # "matplotlib",
    #"seaborn",
    #"plotly",
   # "openai",
    #"sklearn.impute",
    #"sklearn.preprocessing",
    #"sklearn.model_selection",
   # "sklearn.ensemble",
    #"sklearn.metrics",
#    "sklearn.cluster",
 #   "sklearn.svm",
  #  "sklearn.decomposition",
   # "sklearn.manifold",
    #"sklearn.feature_selection",
#    "shap",
 #   "lime",
  #  "lime.lime_tabular",
   # "imblearn.over_sampling",
    #"statsmodels.api",
    #"plotly.graph_objects"
#]
# ///
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import openai
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shap
import plotly.graph_objects as go
from imblearn.over_sampling import SMOTE
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.feature_selection import RFE

# Set your custom proxy URL for the OpenAI API
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
openai_api_key = os.environ.get("AIPROXY_TOKEN")

def perform_generic_analysis(dataframe):
    """
    Perform basic analysis of the dataframe, including summary statistics, missing values, 
    data types, and correlation analysis for numeric columns.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
    tuple: Summary statistics and correlation matrix as dictionaries.
    """
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe(include='all').to_dict()
    }

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    return summary, correlation_matrix

def auto_feature_engineering(dataframe):
    """
    Automatically generate new features such as polynomial features and feature interactions.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
    pd.DataFrame: The enhanced dataframe with new features.
    """
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    
    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(numeric_columns)
    poly_feature_names = poly.get_feature_names_out(numeric_columns.columns)
    
    # Add polynomial features to the dataframe
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
    dataframe = pd.concat([dataframe, poly_df], axis=1)

    return dataframe

def impute_missing_values(dataframe):
    """
    Impute missing values in the dataframe: use median for numerical columns and mode for categorical columns.

    Parameters:
    dataframe (pd.DataFrame): The dataset to impute.

    Returns:
    pd.DataFrame: The dataframe with imputed values.
    """
    # Create imputer for numeric and categorical data
    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    categorical_columns = dataframe.select_dtypes(exclude=[np.number])

    # Impute numerical columns
    dataframe[numeric_columns.columns] = numeric_imputer.fit_transform(numeric_columns)

    # Impute categorical columns
    dataframe[categorical_columns.columns] = categorical_imputer.fit_transform(categorical_columns)

    return dataframe

def perform_time_series_analysis(dataframe, time_column, value_column):
    """
    Perform simple time series analysis using seasonal decomposition.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    time_column (str): The column with time or date information.
    value_column (str): The column containing the values for analysis.

    Returns:
    pd.Series: Decomposed seasonal components of the time series data.
    """
    import statsmodels.api as sm

    # Convert to datetime
    dataframe[time_column] = pd.to_datetime(dataframe[time_column])

    # Seasonal decomposition
    decomposition = sm.tsa.seasonal_decompose(dataframe[value_column], model='additive', period=365)
    decomposition.plot()
    plt.show()

    return decomposition

def hyperparameter_tuning(model, X_train, y_train):
    """
    Perform hyperparameter tuning for model optimization.

    Parameters:
    model: The machine learning model to tune.
    X_train: The training features.
    y_train: The target variable.

    Returns:
    model: The best performing model after hyperparameter tuning.
    """
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

def train_and_evaluate_model(dataframe, target_column):
    """
    Train a model (e.g., Random Forest) and evaluate performance using cross-validation.

    Parameters:
    dataframe (pd.DataFrame): The dataset to train on.
    target_column (str): The column to predict.

    Returns:
    str: Model performance summary.
    """
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]
    
    # Split into train-test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Hyperparameter tuning
    model = hyperparameter_tuning(model, X_train, y_train)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Model evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Confusion Matrix and ROC Curve
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    return {
        'accuracy': accuracy,
        'auc': auc,
        'classification_report': report,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr)
    }

def explain_model_with_lime(model, X_train, y_train):
    """
    Use LIME to explain model predictions.

    Parameters:
    model: The trained model.
    X_train: The training dataset.
    y_train: The target variable.

    Returns:
    lime_explainer: LIME explanations.
    """
    explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns, class_names=[str(i) for i in np.unique(y_train)], discretize_continuous=True)
    lime_explanations = explainer.explain_instance(X_train.iloc[0].values, model.predict_proba)
    
    # Display explanation
    lime_explanations.show_in_notebook()

    return lime_explanations

def ensemble_model_analysis(X_train, y_train):
    """
    Combine multiple models using Voting and Stacking classifiers.

    Parameters:
    X_train: The training features.
    y_train: The target variable.

    Returns:
    ensemble_model: The ensemble model.
    """
    # Base models
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    lr = LogisticRegression(max_iter=1000)
    gb = GradientBoostingClassifier()

    # Stacking Classifier
    stack_model = StackingClassifier(estimators=[('rf', rf), ('lr', lr), ('gb', gb)], final_estimator=LogisticRegression())
    stack_model.fit(X_train, y_train)
    
    # Voting Classifier
    voting_model = VotingClassifier(estimators=[('rf', rf), ('lr', lr), ('gb', gb)], voting='soft')
    voting_model.fit(X_train, y_train)

    return stack_model, voting_model

def generate_interactive_report(data_summary, analysis_results, charts, advanced_analysis_results, model_evaluation_results):
    """
    Generate an interactive report with detailed analysis and visualizations.

    Parameters:
    data_summary (dict): Summary statistics of the dataset.
    analysis_results (dict): Results of various analyses.
    charts (list): Visualization charts.
    advanced_analysis_results (dict): Advanced analyses results.
    model_evaluation_results (dict): Model evaluation results.

    Returns:
    str: Generated interactive report.
    """
    report = f"""
    ### Data Summary:
    {data_summary}

    ### Analysis Results:
    {analysis_results}

    ### Advanced Analysis:
    {advanced_analysis_results}

    ### Model Evaluation:
    {model_evaluation_results}

    ### Visualizations:
    {charts}
    """
    
    return report

# Example usage:
# dataframe = pd.read_csv("your_data.csv")
# summary, correlation_matrix = perform_generic_analysis(dataframe)
# dataframe = auto_feature_engineering(dataframe)
# dataframe = impute_missing_values(dataframe)
# perform_time_series_analysis(dataframe, 'Date', 'Sales')
# model_results = train_and_evaluate_model(dataframe, target_column='target')
# lime_explanation = explain_model_with_lime(model, X_train, y_train)
# stack_model, voting_model = ensemble_model_analysis(X_train, y_train)
# interactive_report = generate_interactive_report(summary, {'correlation_matrix': correlation_matrix}, charts, advanced_analysis_results, model_results)
# print(interactive_report)


def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.

    This function follows a complete data analysis pipeline, including:
    1. Loading the dataset.
    2. Performing basic and advanced analyses.
    3. Visualizing results.
    4. Training machine learning models.
    5. Generating an interactive report summarizing all results.

    Parameters:
    input_file (str): Path to the CSV file to be analyzed.
    """
    # Load the CSV data into a pandas DataFrame
    dataframe = pd.read_csv(input_file, encoding='latin1')  # Loading the dataset

    # Perform generic analysis like summary statistics and correlation matrix
    summary, correlation_matrix = perform_generic_analysis(dataframe)

    # Auto-feature engineering: automating the process of creating new features and preparing data
    dataframe = auto_feature_engineering(dataframe)

    # Handle missing values by applying imputation techniques (e.g., mean, median, etc.)
    dataframe = impute_missing_values(dataframe)

    # Perform time series analysis to identify trends, seasonality, and other time-dependent features
    time_series_results = perform_time_series_analysis(dataframe, 'Date', 'Sales')

    # Train and evaluate a machine learning model on the dataset, using a target column for prediction
    model_results = train_and_evaluate_model(dataframe, target_column='target')

    # Use LIME (Local Interpretable Model-agnostic Explanations) to explain the model's predictions
    lime_explanation = explain_model_with_lime(model_results['model'], dataframe.drop(columns='target'), dataframe['target'])

    # Apply ensemble learning techniques (stacking and voting) to combine multiple models for better accuracy
    stack_model, voting_model = ensemble_model_analysis(dataframe.drop(columns='target'), dataframe['target'])

    # Create visualizations like histograms, heatmaps, and charts based on the analysis
    charts = create_histograms(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    # Collect advanced analysis results such as time series trends, clustering, and outliers
    advanced_analysis_results = {
        'time_series_analysis': time_series_results,
        'outlier_detection': detect_outliers_zscore(dataframe)[1],  # Assuming outlier detection function returns the results
        'cluster_analysis': dynamic_cluster_analysis(dataframe)
    }

    # Generate an interactive report by integrating the analysis, model results, and charts
    interactive_report = generate_interactive_report(summary, {'correlation_matrix': correlation_matrix}, charts, advanced_analysis_results, model_results)

    # Output the interactive report (or you could save it to a file, depending on the use case)
    print(interactive_report)

# Entry point of the script when run from the command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    # Run the CSV analysis
    analyze_csv(input_file)









