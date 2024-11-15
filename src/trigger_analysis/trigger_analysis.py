import pandas as pd
from src.api_interface import get_openai_response, get_dual_responses
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
from src.stylometric_analysis import calculate_stylometric_similarity

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

def calculate_thematic_similarity(response, reference_text):
    """Calculate thematic similarity using TF-IDF or other thematic features."""
    embeddings = model.encode([response, reference_text])
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_similarity

def run_trigger_analysis(canary_data, prompt_column):
    """Runs trigger analysis on the provided canary data for the specified prompt column."""
    results = []
    for index, row in canary_data.iterrows():
        prompt = row[prompt_column]
        expected_response = row['Expected Response']
        category = row['Category']
        
        # Get responses from both models
        responses = get_dual_responses(prompt)
        
        # Calculate similarity scores for OpenAI
        openai_lev_sim, openai_cos_sim = calculate_similarity(expected_response, responses['openai'])
        
        # Calculate similarity scores for Gemini
        gemini_lev_sim, gemini_cos_sim = calculate_similarity(expected_response, responses['gemini'])
        
        # Calculate additional thematic and stylometric similarity
        openai_stylometric_sim = calculate_stylometric_similarity(expected_response, responses['openai'])
        gemini_stylometric_sim = calculate_stylometric_similarity(expected_response, responses['gemini'])

        openai_thematic_sim = calculate_thematic_similarity(responses['openai'], expected_response)
        gemini_thematic_sim = calculate_thematic_similarity(responses['gemini'], expected_response)
        
        results.append({
            'prompt': prompt,
            'expected_response': expected_response,
            'openai_response': responses['openai'],
            'gemini_response': responses['gemini'],
            'category': category,
            'openai_levenshtein_similarity': openai_lev_sim,
            'openai_cosine_similarity': openai_cos_sim,
            'gemini_levenshtein_similarity': gemini_lev_sim,
            'gemini_cosine_similarity': gemini_cos_sim,
            'openai_stylometric_similarity': openai_stylometric_sim,
            'gemini_stylometric_similarity': gemini_stylometric_sim,
            'openai_thematic_similarity': openai_thematic_sim,
            'gemini_thematic_similarity': gemini_thematic_sim
        })
    
    return pd.DataFrame(results)

def analyze_rephrased_prompts():
    """Compares the original and rephrased prompts to assess impact on model responses."""
    # Load canary data containing both original and rephrased prompts
    canary_data = load_canary_data('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/canary_data/rephrased_canaries.csv')

    # Run analysis on original prompts
    original_results = run_trigger_analysis(canary_data, prompt_column='Original Prompt')
    original_results.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv', index=False)

    # Run analysis on rephrased prompts
    rephrased_results = run_trigger_analysis(canary_data, prompt_column='Rephrased Prompt')
    rephrased_results.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results_rephrased.csv', index=False)

    # Merge results for comparison
    comparison_df = original_results.merge(
        rephrased_results,
        on=['expected_response', 'category'],
        suffixes=('_original', '_rephrased')
    )
    
    # Calculate differences in similarity scores for both models
    comparison_df['openai_levenshtein_diff'] = (
        comparison_df['openai_levenshtein_similarity_rephrased'] - 
        comparison_df['openai_levenshtein_similarity_original']
    )
    comparison_df['openai_cosine_diff'] = (
        comparison_df['openai_cosine_similarity_rephrased'] - 
        comparison_df['openai_cosine_similarity_original']
    )
    comparison_df['gemini_levenshtein_diff'] = (
        comparison_df['gemini_levenshtein_similarity_rephrased'] - 
        comparison_df['gemini_levenshtein_similarity_original']
    )
    comparison_df['gemini_cosine_diff'] = (
        comparison_df['gemini_cosine_similarity_rephrased'] - 
        comparison_df['gemini_cosine_similarity_original']
    )
    
    # Save the comparison results
    comparison_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/rephrasing_impact_summary.csv', index=False)

if __name__ == "__main__":
    # Run trigger analysis for both original and rephrased prompts
    analyze_rephrased_prompts()