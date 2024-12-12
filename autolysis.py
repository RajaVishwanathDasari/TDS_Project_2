import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests  # Using requests for custom API call to the proxy
import sys
from pathlib import Path
import json

# Set your custom proxy URL for the OpenAI API
openai_api_base = "https://aiproxy.sanand.workers.dev/openai/"

# Load your AIPROXY_TOKEN environment variable (or set it directly)
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")

# Define headers for the API requests
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

def generate_story(data_summary, analysis_results, charts):
    # Converting pandas data types to standard Python types to avoid issues with json serialization
    data_summary = {
        'columns': list(data_summary['columns']),
        'data_types': dict(data_summary['data_types']),
        'missing_values': dict(data_summary['missing_values']),
        'summary_statistics': {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in data_summary['summary_statistics'].items()}
    }

    # Convert analysis results to standard Python types
    analysis_results = {
        'correlation_matrix': analysis_results.get('correlation_matrix', None),
        'outliers': analysis_results.get('outliers', None)
    }

    # Convert charts to a simple list of strings (file paths)
    charts = [str(chart) for chart in charts]

    prompt = f"""
    Write a detailed story about the analysis of a dataset. The story should describe:
    1. A brief summary of the data: what the data consists of, column names, types, and any relevant patterns or issues.
    2. The analysis performed: including steps like correlation, outlier detection, and any other relevant analysis.
    3. Insights discovered: any patterns, anomalies, trends, or key takeaways from the data.
    4. Implications: what actions can be taken based on the insights? What do these findings mean for decision-making?
    
    Data Summary: {data_summary}
    
    Analysis Results:
    {analysis_results}

    Charts: {charts}
    """
    
    data = {
        "model": "gpt-4o-mini",  # Example model (adjust as necessary)
        "messages": [{"role": "user", "content": prompt}]
    }
    
    # Ensure the URL path is correctly concatenated
    api_url = f"{openai_api_base}v1/chat/completions"
    
    # Make the API request
    response = requests.post(
        api_url,  # Using the corrected URL
        headers=headers,
        data=json.dumps(data)
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"Error generating story: {response.text}")


def perform_generic_analysis(dataframe):
    # Summarize dataset: column names, types, missing values, summary statistics
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe()
    }
    
    # Only select numeric columns for correlation and outlier detection
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    
    # Correlation matrix (only on numeric columns)
    correlation_matrix = numeric_columns.corr() if numeric_columns.shape[1] > 1 else None

    # Outlier detection using IQR for numeric columns
    if numeric_columns.shape[1] > 0:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).sum()
    else:
        outliers = None

    return summary, correlation_matrix, outliers

def create_histograms(dataframe, numerical_cols):
    # Save histograms as PNG
    charts = []
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(dataframe[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        chart_path = f"{col}_histogram.png"
        plt.savefig(chart_path, dpi=100)
        plt.close()
        charts.append(chart_path)
    return charts

def create_boxplots(dataframe, numerical_cols):
    # Boxplot for outlier detection
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataframe[numerical_cols])
    plt.title('Outlier Detection - Boxplot')
    boxplot_path = 'outliers_boxplot.png'
    plt.savefig(boxplot_path, dpi=100)
    plt.close()
    return boxplot_path

def create_correlation_heatmap(correlation_matrix):
    # Correlation heatmap
    if correlation_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_path = 'correlation_matrix.png'
        plt.savefig(heatmap_path, dpi=100)
        plt.close()
        return heatmap_path
    return None

def create_readme(ai_story, charts, summary):
    with open('README.md', 'w') as readme_file:
        readme_file.write("# Automated Data Analysis Report\n\n")
        readme_file.write("## Dataset Summary\n")
        readme_file.write(f"Columns: {summary['columns']}\n")
        readme_file.write(f"Data Types: {summary['data_types']}\n")
        readme_file.write(f"Missing Values: {summary['missing_values']}\n")
        readme_file.write("Summary Statistics:\n")
        readme_file.write(f"{summary['summary_statistics']}\n\n")

        readme_file.write("## AI-Generated Story\n")
        readme_file.write(ai_story + "\n\n")
        
        readme_file.write("## Data Visualizations\n")
        for chart in charts:
            readme_file.write(f"![{chart}]({chart})\n")

def analyze_csv(input_file):
    # Load dataset
    dataframe = pd.read_csv(input_file, encoding = 'latin1')

    # Step 1: Perform generic analysis
    summary, correlation_matrix, outliers = perform_generic_analysis(dataframe)

    # Step 2: AI-generated story
    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix, 'outliers': outliers}, [])

    # Step 3: Create visualizations (e.g., histograms, outliers, correlation matrix)
    numerical_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns
    charts = []

    # Create histograms for numerical columns
    charts.extend(create_histograms(dataframe, numerical_cols))

    # Create boxplot for outliers
    charts.append(create_boxplots(dataframe, numerical_cols))

    # Create correlation heatmap
    if correlation_matrix is not None:
        charts.append(create_correlation_heatmap(correlation_matrix))

    # Step 4: Create README file with analysis summary, AI insights, and charts
    create_readme(ai_story, charts, summary)
    print("Analysis complete. Check README.md and chart files.")

if __name__ == "__main__":
    # Ensure CSV filename is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)
