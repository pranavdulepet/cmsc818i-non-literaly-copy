# src/trigger_analysis.py

import pandas as pd
from api_interface import get_openai_response
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_canary_data(file_path):
    """Loads canary data from the specified file path."""
    return pd.read_csv(file_path)

def calculate_similarity(expected, response):
    """Calculates Levenshtein and cosine similarity between expected and model responses."""
    # Levenshtein similarity (normalized)
    lev_sim = 1 - (levenshtein_distance(expected, response) / max(len(expected), len(response)))
    
    # Cosine similarity
    embeddings = model.encode([expected, response])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return lev_sim, cos_sim

def run_trigger_analysis(canary_data, prompt_column):
    """Runs trigger analysis on the provided canary data for the specified prompt column."""
    results = []
    for index, row in canary_data.iterrows():
        prompt = row[prompt_column]
        expected_response = row['Expected Response']
        category = row['Category']
        model_response = get_openai_response(prompt)
        
        # Calculate similarity scores
        lev_sim, cos_sim = calculate_similarity(expected_response, model_response)
        
        results.append({
            'prompt': prompt,
            'expected_response': expected_response,
            'model_response': model_response,
            'category': category,
            'levenshtein_similarity': lev_sim,
            'cosine_similarity': cos_sim
        })
    
    return pd.DataFrame(results)

def analyze_rephrased_prompts():
    """Compares the original and rephrased prompts to assess impact on model responses."""
    # Load canary data containing both original and rephrased prompts
    canary_data = load_canary_data('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/canary_data/rephrased_canaries.csv')

    # Run analysis on original prompts
    original_results = run_trigger_analysis(canary_data, prompt_column='Original Prompt')
    original_results.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results_original.csv', index=False)

    # Run analysis on rephrased prompts
    rephrased_results = run_trigger_analysis(canary_data, prompt_column='Rephrased Prompt')
    rephrased_results.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results_rephrased.csv', index=False)

    # Merge results for comparison
    comparison_df = original_results.merge(rephrased_results, on=['expected_response', 'category'], suffixes=('_original', '_rephrased'))
    
    # Calculate differences in similarity scores
    comparison_df['levenshtein_diff'] = comparison_df['levenshtein_similarity_rephrased'] - comparison_df['levenshtein_similarity_original']
    comparison_df['cosine_diff'] = comparison_df['cosine_similarity_rephrased'] - comparison_df['cosine_similarity_original']
    
    # Save the comparison results
    comparison_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/rephrasing_impact_summary.csv', index=False)

if __name__ == "__main__":
    # Run trigger analysis for both original and rephrased prompts
    analyze_rephrased_prompts()