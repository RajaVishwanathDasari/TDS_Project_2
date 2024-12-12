import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from datetime import datetime
import json

# Initialize OpenAI API
openai.api_key = 'your-openai-api-key'

def load_data(file_path):
    """Load and clean data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        data = data.dropna()  # Basic data cleaning
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_with_llm(data):
    """Send data analysis request to OpenAI GPT-3 model."""
    # Convert the data to a JSON format or any other structure that works
    data_json = data.head().to_json(orient='records')
    
    prompt = f"""
    Analyze the following dataset and provide insights, correlations, and potential trends:
    {data_json}
    """

    try:
        # Request LLM for analysis
        response = openai.Completion.create(
            model="gpt-4",  # Or another model like "gpt-3.5-turbo"
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        analysis_results = response.choices[0].text.strip()
        return analysis_results
    except Exception as e:
        print(f"Error in LLM request: {e}")
        return "Analysis not available."

def visualize_data(data):
    """Generate and save visualizations."""
    plt.figure(figsize=(10, 8))

    # Generate a correlation heatmap
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    print("Correlation matrix saved.")

    # Generate a histogram of the first column (for demonstration)
    data.iloc[:, 0].hist(bins=20, alpha=0.75)
    plt.title(f"Histogram of {data.columns[0]}")
    plt.xlabel(data.columns[0])
    plt.ylabel("Frequency")
    plt.savefig("histogram.png")
    print("Histogram saved.")

def generate_narrative(data, analysis_results):
    """Generate a markdown narrative for the analysis."""
    narrative = "# Data Analysis Report\n\n"
    narrative += "## Descriptive Statistics\n"
    narrative += data.describe().to_markdown() + "\n"
    
    narrative += "## LLM Insights\n"
    narrative += analysis_results + "\n"

    # Adding some basic visualizations as images
    narrative += "## Visualizations\n"
    narrative += "### Correlation Matrix\n![Correlation Matrix](correlation_matrix.png)\n"
    narrative += "### Histogram\n![Histogram](histogram.png)\n"

    return narrative

def save_results(narrative):
    """Save the narrative and visualizations to disk."""
    with open("analysis_report.md", "w") as f:
        f.write(narrative)
    print("Analysis report saved as 'analysis_report.md'.")

def run_analysis(file_path):
    """Main function to load, analyze, visualize, and generate report."""
    # Load data
    data = load_data(file_path)
    if data is None:
        return
    
    # Send the data for analysis to the LLM
    analysis_results = analyze_with_llm(data)
    
    # Visualize the data
    visualize_data(data)
    
    # Generate a comprehensive narrative
    narrative = generate_narrative(data, analysis_results)
    
    # Save everything
    save_results(narrative)

if __name__ == "__main__":
    # Set the file path to the dataset
    dataset_path = os.path.join(os.getcwd(), "datasets", "your_dataset.csv")
    run_analysis(dataset_path)
