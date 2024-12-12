import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import openai

# Setup OpenAI API
def setup_openai_api():
    try:
        api_token = os.environ["AIPROXY_TOKEN"]
        openai.api_key = api_token
    except KeyError:
        print("Error: AIPROXY_TOKEN environment variable not set.")
        sys.exit(1)

# Utility function to save and return chart path
def save_chart(fig, filename):
    path = f"{filename}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path

# Generate a summary of the dataset
def dataset_summary(df):
    summary = {
        "shape": df.shape,
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_statistics": df.describe(include='all').to_dict()
    }
    return summary

# Generate visualizations
def generate_visualizations(df):
    charts = []
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        corr = df.corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        charts.append(save_chart(fig, "correlation_matrix"))

    if len(df.select_dtypes(include=[np.number]).columns) > 0:
        for col in df.select_dtypes(include=[np.number]).columns[:3]:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            charts.append(save_chart(fig, f"distribution_{col}"))

    return charts

# Interact with LLM for analysis and narrative
def interact_with_llm(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data analysis assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error interacting with LLM: {e}"

# Main script execution
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    if not os.path.exists(dataset_path):
        print(f"Error: File {dataset_path} does not exist.")
        sys.exit(1)

    setup_openai_api()

    df = pd.read_csv(dataset_path)
    summary = dataset_summary(df)
    charts = generate_visualizations(df)

    llm_prompt = (
        f"Dataset Analysis:\n\n"
        f"Shape: {summary['shape']}\n\n"
        f"Columns and Types: {summary['columns']}\n\n"
        f"Missing Values: {summary['missing_values']}\n\n"
        f"Summary Statistics: {summary['summary_statistics']}\n\n"
        f"Generate a story summarizing the dataset and providing insights based on this information. "
        f"Refer to the generated visualizations as supporting evidence."
    )

    narrative = interact_with_llm(llm_prompt)

    readme_content = f"""# Automated Analysis Report

## Dataset Summary

- **Shape:** {summary['shape']}
- **Columns:** {summary['columns']}
- **Missing Values:** {summary['missing_values']}

## Insights and Narrative

{narrative}

## Visualizations

"""

    for chart in charts:
        readme_content += f"![{chart}]({chart})\n\n"

    with open("README.md", "w") as f:
        f.write(readme_content)

    print("Analysis complete. Results saved to README.md and PNG files.")

if __name__ == "__main__":
    main()
