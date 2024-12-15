# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "requests",
#   "pathlib",
#   "matplotlib"
# ]
# ///
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import json
import requests
import sys
from statsmodels.tsa.arima.model import ARIMA

# Set your custom proxy URL for the OpenAI API
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Load your AIPROXY_TOKEN environment variable
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

    # Correlation matrix for numeric columns
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    return summary, correlation_matrix


def detect_outliers_z_score(dataframe, threshold=3):
    """
    Detect outliers in numeric columns using the Z-score method.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    threshold (float): The Z-score threshold to identify outliers.

    Returns:
    tuple: Updated dataframe with outlier column and a summary of detected outliers.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for outlier detection.")
        return dataframe, {}

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outlier_summary = (z_scores > threshold).sum(axis=0).to_dict()
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)

    return dataframe, outlier_summary


def perform_time_series_analysis(dataframe, target_column, date_column):
    """
    Perform time series forecasting using ARIMA if time-related data is detected.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    target_column (str): The column to forecast.
    date_column (str): The column representing dates.

    Returns:
    dict: Forecast results and ARIMA model summary.
    """
    try:
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        dataframe.set_index(date_column, inplace=True)
        ts_data = dataframe[target_column].dropna()

        # Fit ARIMA model
        model = ARIMA(ts_data, order=(1, 1, 1))
        fitted_model = model.fit()
        forecast = fitted_model.forecast(steps=10)

        return {
            'forecast': forecast.tolist(),
            'model_summary': fitted_model.summary().as_text()
        }
    except Exception as e:
        print(f"Time series analysis failed: {e}")
        return None


def dynamic_cluster_analysis(dataframe, max_clusters=5):
    """
    Perform dynamic clustering analysis using K-means, with the number of clusters determined by data complexity.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    max_clusters (int): Maximum number of clusters.

    Returns:
    pd.DataFrame: DataFrame with assigned clusters.
    """
    from sklearn.cluster import KMeans

    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for clustering.")
        return None

    # Normalize data
    scaled_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

    # Determine optimal clusters dynamically (Elbow method suggestion could be implemented here)
    kmeans = KMeans(n_clusters=min(len(scaled_data), max_clusters), random_state=42)
    dataframe['cluster'] = kmeans.fit_predict(scaled_data)

    return dataframe[['cluster']]


def create_histograms(dataframe, bins=10):
    """
    Create histograms for numeric columns with labels and annotations.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    bins (int): Number of bins for histograms.

    Returns:
    list: Matplotlib figures for histograms.
    """
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.histplot(dataframe[col], bins=bins, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate significant points
        plt.annotate(f"Mean: {dataframe[col].mean():.2f}", xy=(0.7, 0.9), xycoords='axes fraction')
        plt.annotate(f"Max: {dataframe[col].max():.2f}", xy=(0.7, 0.85), xycoords='axes fraction')
        histograms.append(plt.gcf())
        plt.close()

    return histograms


def create_correlation_heatmap(dataframe):
    """
    Create a labeled correlation heatmap for numeric columns.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
    Matplotlib figure: Correlation heatmap.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Add key insights as annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                plt.text(j, i, "High Corr", ha='center', va='center', color='red', fontsize=9)

    plt.tight_layout()
    return plt.gcf()


def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a detailed narrative using LLM API based on dataset analysis, with iterative refinements.

    Parameters:
    data_summary (dict): Summary statistics of the dataset.
    analysis_results (dict): Results of various analyses.
    charts (list): Visualization charts.
    advanced_analysis_results (dict): Advanced analyses results.

    Returns:
    str: Generated narrative.
    """
    prompt = f"""
    Write a comprehensive report:
    1. Dataset summary: {data_summary}
    2. Analysis insights: {analysis_results}
    3. Advanced results: {advanced_analysis_results}
    4. Charts: Describe key observations from visualizations.
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700
    }

    response = requests.post(openai_api_url, headers=headers, json=data)
    if response.status_code == 200:
        narrative = response.json().get('choices', [{}])[0].get('message', {}).get('content', "AI generation failed.")

        # Iterative refinement if key details are missing
        if "details missing" in narrative.lower():
            data["messages"].append({"role": "user", "content": "Refine and expand on missing details."})
            response = requests.post(openai_api_url, headers=headers, json=data)
            if response.status_code == 200:
                narrative = response.json().get('choices', [{}])[0].get('message', {}).get('content', "AI generation failed.")

        return narrative
    else:
        print(f"Error: {response.status_code}\n{response.text}")
        return "AI generation failed."


def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.

    Parameters:
    input_file (str): Path to the CSV file.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Perform analysis
    summary, correlation_matrix = perform_generic_analysis(dataframe)
    dataframe, outlier_summary = detect_outliers_z_score(dataframe)

    advanced_analysis_results = {
        'time_series_forecast': perform_time_series_analysis(dataframe, 'target_column', 'date_column'),
        'cluster_analysis': dynamic_cluster_analysis(dataframe),
        'outlier_detection': outlier_summary
    }

    # Create visualizations
    charts = create_histograms(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    # Generate narrative
    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix}, charts, advanced_analysis_results)

    print(ai_story)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)


    analyze_csv(input_file)



