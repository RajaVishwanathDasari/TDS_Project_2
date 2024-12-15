import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import sys
from pathlib import Path
import json

# Set your custom proxy URL for the OpenAI API
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Load your AIPROXY_TOKEN environment variable
openai_api_key = os.environ.get("AIPROXY_TOKEN")

def summarize_statistics(summary_statistics):
    """Summarize the dataset's statistics, focusing on key metrics."""
    summarized_stats = {}
    for column, stats in summary_statistics.items():
        summarized_stats[column] = {
            'mean': stats.get('mean'),
            'std': stats.get('std'),
            'min': stats.get('min'),
            'max': stats.get('max')
        }
    return summarized_stats

def filter_correlation_matrix(correlation_matrix, threshold=0.8):
    """Filter out low-correlation pairs to focus on significant relationships."""
    filtered_correlation = {}
    for col, correlations in correlation_matrix.items():
        high_corr = {k: v for k, v in correlations.items() if abs(v) > threshold}
        if high_corr:
            filtered_correlation[col] = high_corr
    return filtered_correlation

def generate_dynamic_prompt(data_summary, analysis_results, charts, previous_responses=None):
    """Generate a dynamic prompt that incorporates previous responses and analysis."""
    data_summary = {
        'columns': list(data_summary['columns']),
        'data_types': {k: str(v) for k, v in data_summary['data_types'].items()},
        'missing_values': data_summary['missing_values'],
        'summary_statistics': summarize_statistics(data_summary['summary_statistics'])  # Reduced stats
    }

    analysis_results = {
        'correlation_matrix': filter_correlation_matrix(analysis_results.get('correlation_matrix', {})),  # Filtered correlations
        'outliers': analysis_results.get('outliers', None)
    }

    charts = [str(chart) for chart in charts]

    prompt = f"""
    Write a comprehensive analysis report based on the following information:
    1. Dataset summary: columns, data types, missing values, and summary statistics.
    2. Analytical insights: correlation matrix, outlier details, and any patterns or anomalies.
    3. Visualizations: include descriptions of the provided charts.
    
    **Explanation of Techniques Applied:**
    - Summary statistics: Provides a quick snapshot of the dataset, including mean, standard deviation, and range of values.
    - Correlation analysis: Used to identify potential relationships between numerical variables, guiding feature selection for predictive models.
    - Outlier detection: Identifies extreme values that might distort analysis or indicate important anomalies.
    - Visualizations: Histograms and boxplots provide an intuitive way to understand data distributions and detect outliers visually.

    Data Summary: {data_summary}
    Analysis Results: {analysis_results}
    Charts: {charts}
    """

    if previous_responses:
        prompt = "\n".join([previous_responses, prompt])

    return prompt

def generate_story(data_summary, analysis_results, charts):
    """Generate a detailed story using the AI API based on the dataset analysis."""
    prompt = generate_dynamic_prompt(data_summary, analysis_results, charts)

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700
    }

    response = requests.post(openai_api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "AI generation failed.")
    else:
        print(f"Error: {response.status_code}\n{response.text}")
        return "AI generation failed."

def perform_generic_analysis(dataframe):
    """Perform basic analysis including summary statistics and outlier detection."""
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe().to_dict()
    }

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    # Outlier detection using Interquartile Range (IQR)
    if not numeric_columns.empty:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).sum().to_dict()
    else:
        outliers = None

    return summary, correlation_matrix, outliers

def create_histograms(dataframe, numerical_cols):
    """Generate and save histograms for numerical columns."""
    charts = []
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(dataframe[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        chart_path = f"{col}_histogram.png"
        plt.savefig(chart_path, dpi=100)
        plt.close()
        charts.append(chart_path)
    return charts

def create_boxplots(dataframe, numerical_cols):
    """Generate a boxplot for numerical columns to detect outliers."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe[numerical_cols], orient='h')
    plt.title('Boxplot for Outlier Detection')
    plt.xlabel('Values')
    boxplot_path = 'outliers_boxplot.png'
    plt.savefig(boxplot_path, dpi=100)
    plt.close()
    return boxplot_path

def create_correlation_heatmap(correlation_matrix):
    """Generate a heatmap for the correlation matrix."""
    if correlation_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(correlation_matrix), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        heatmap_path = 'correlation_matrix.png'
        plt.savefig(heatmap_path, dpi=100)
        plt.close()
        return heatmap_path
    return None

def create_readme(ai_story, charts, summary):
    """Generate a README file summarizing the analysis."""
    with open('README.md', 'w', encoding='utf-8') as readme_file:
        readme_file.write("# Automated Data Analysis Report\n\n")
        readme_file.write("## Dataset Summary\n")
        readme_file.write(f"### Columns: {summary['columns']}\n")
        readme_file.write(f"### Data Types: {summary['data_types']}\n")
        readme_file.write(f"### Missing Values: {summary['missing_values']}\n")
        readme_file.write("### Summary Statistics:\n")
        readme_file.write(json.dumps(summary['summary_statistics'], indent=4))

        readme_file.write("\n\n## AI-Generated Insights\n")
        readme_file.write(ai_story + "\n\n")

        readme_file.write("## Data Visualizations\n")
        for chart in charts:
            readme_file.write(f"![{chart}]({chart})\n")

def analyze_csv(input_file):
    """Main function to perform the analysis on the provided CSV file."""
    dataframe = pd.read_csv(input_file, encoding='latin1')

    summary, correlation_matrix, outliers = perform_generic_analysis(dataframe)

    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix, 'outliers': outliers}, [])

    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns

    charts = create_histograms(dataframe, numerical_cols)
    charts.append(create_boxplots(dataframe, numerical_cols))
    if correlation_matrix:
        charts.append(create_correlation_heatmap(correlation_matrix))

    create_readme(ai_story, charts, summary)
    print("Analysis complete. Check README.md and chart files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)
