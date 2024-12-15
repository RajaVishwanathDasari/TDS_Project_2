# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "requests",
#   "pathlib",
#   "openai",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn"
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

# Set your custom proxy URL for the OpenAI API
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Load your AIPROXY_TOKEN environment variable
openai_api_key = os.environ.get("AIPROXY_TOKEN")

def perform_generic_analysis(dataframe):
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe(include='all').to_dict()
    }

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    return summary, correlation_matrix

def detect_outliers_zscore(dataframe, threshold=3):
    numeric_data = dataframe.select_dtypes(include=[np.number])
    if numeric_data.empty:
        print("No numeric columns for outlier detection.")
        return dataframe, {}

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outlier_summary = (z_scores > threshold).sum(axis=0).to_dict()
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)

    return dataframe, outlier_summary

def perform_time_series_analysis(dataframe, target_column, date_column):
    try:
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        dataframe.sort_values(by=date_column, inplace=True)

        dataframe['year_month'] = dataframe[date_column].dt.to_period('M')
        grouped = dataframe.groupby('year_month')[target_column].mean().reset_index()

        trend = "increasing" if grouped[target_column].iloc[-1] > grouped[target_column].iloc[0] else "decreasing"

        return {
            'trend': trend,
            'monthly_averages': grouped.to_dict(orient='records')
        }
    except Exception as e:
        print(f"Time series analysis failed: {e}")
        return None

def dynamic_cluster_analysis(dataframe, max_clusters=5):
    from sklearn.cluster import KMeans

    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for clustering.")
        return None

    scaled_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

    kmeans = KMeans(n_clusters=min(len(scaled_data), max_clusters), random_state=42)
    dataframe['cluster'] = kmeans.fit_predict(scaled_data)

    return dataframe[['cluster']]

def create_histograms(dataframe, bins=10):
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.histplot(dataframe[col], bins=bins, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        plt.annotate(f"Mean: {dataframe[col].mean():.2f}", xy=(0.7, 0.9), xycoords='axes fraction')
        plt.annotate(f"Max: {dataframe[col].max():.2f}", xy=(0.7, 0.85), xycoords='axes fraction')
        histograms.append(plt.gcf())
        plt.close()

    return histograms

def create_correlation_heatmap(dataframe):
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                plt.text(j, i, "High Corr", ha='center', va='center', color='red', fontsize=9)

    plt.tight_layout()
    return plt.gcf()

import openai

def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a concise narrative using LLM API based on dataset analysis, with compact summaries for token efficiency.

    Parameters:
    data_summary (dict): Summary statistics of the dataset.
    analysis_results (dict): Results of various analyses.
    charts (list): Visualization charts.
    advanced_analysis_results (dict): Advanced analyses results.

    Returns:
    str: Generated narrative.
    """
    # Summarize the results before sending to the API for token efficiency
    summary_text = f"Dataset summary:\nColumns: {data_summary['columns']}\nMissing values: {data_summary['missing_values']}\n"
    summary_text += f"Data types: {data_summary['data_types']}\nStatistics: {json.dumps(data_summary['summary_statistics'], indent=2)}"

    analysis_text = f"\nCorrelation matrix: {json.dumps(analysis_results.get('correlation_matrix', {}), indent=2)}"
    outliers_text = f"\nOutliers: {json.dumps(analysis_results.get('outlier_summary', {}), indent=2)}"

    advanced_text = f"\nAdvanced Analysis Results:\nTime Series: {json.dumps(advanced_analysis_results.get('time_series_analysis', {}), indent=2)}"
    advanced_text += f"\nClusters: {json.dumps(advanced_analysis_results.get('cluster_analysis', {}), indent=2)}"

    charts_info = f"\nCharts Observations: {str([str(chart) for chart in charts])}"

    # Prepare the request content
    prompt = f"""
    You are a data analysis assistant. Generate a comprehensive yet concise report based on the following:
    {summary_text}
    {analysis_text}
    {outliers_text}
    {advanced_text}
    {charts_info}
    """

    # Request to OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an assistant that helps generate detailed reports from data analysis."},
                  {"role": "user", "content": prompt}],
    )

    if response.status_code == 200:
        narrative = response.choices[0].message['content']
        return narrative
    else:
        print(f"Error: {response.status_code}\n{response.text}")
        return "AI generation failed."

# Example of integrating the above function into your existing workflow:
summary, correlation_matrix = perform_generic_analysis(dataframe)
advanced_analysis_results = {
    'time_series_analysis': perform_time_series_analysis(dataframe, 'target_column', 'date_column'),
    'cluster_analysis': dynamic_cluster_analysis(dataframe),
    'outlier_detection': outlier_summary
}
charts = create_histograms(dataframe)
charts.append(create_correlation_heatmap(dataframe))

ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix}, charts, advanced_analysis_results)
print(ai_story)

def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.

    Parameters:
    input_file (str): Path to the CSV file.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Perform analysis
    summary, correlation_matrix = perform_generic_analysis(dataframe)
    dataframe, outlier_summary = detect_outliers_zscore(dataframe)

    advanced_analysis_results = {
        'time_series_analysis': perform_time_series_analysis(dataframe, 'target_column', 'date_column'),
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








