import pandas as pd
from src.api_interface import get_openai_response, get_dual_responses
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
from src.stylometric_analysis import calculate_stylometric_similarity
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(expected, response):
    """Calculates Levenshtein and cosine similarity between expected and model responses."""
    # Levenshtein similarity (normalized)
    lev_sim = 1 - (levenshtein_distance(expected, response) / max(len(expected), len(response)))
    
    # Cosine similarity
    embeddings = model.encode([expected, response])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return lev_sim, cos_sim

def calculate_thematic_similarity(response, reference):
    """Calculates thematic similarity between two responses."""
    embeddings = model.encode([response, reference])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def generate_repeated_prompt(phrase, repetitions):
    """Creates a prompt that simulates repeated exposure by embedding the phrase multiple times."""
    return " ".join([phrase] * repetitions)

def run_memorization_test(fine_tune_level, phrases_df):
    results = []
    
    for _, row in phrases_df.iterrows():
        phrase = row['phrase']
        expected_response = row['expected_response']
        category = row['category']
        
        # Create prompt with repeated phrases to simulate exposure
        prompt = generate_repeated_prompt(phrase, fine_tune_level)
        
        # Get responses from both models
        responses = get_dual_responses(prompt)
        
        # Calculate similarity scores for OpenAI
        openai_lev_sim, openai_cos_sim = calculate_similarity(expected_response, responses['openai'])
        gemini_lev_sim, gemini_cos_sim = calculate_similarity(expected_response, responses['gemini'])
        
        # Calculate additional thematic and stylometric similarity for non-literal analysis
        openai_thematic_sim = calculate_thematic_similarity(responses['openai'], expected_response)
        gemini_thematic_sim = calculate_thematic_similarity(responses['gemini'], expected_response)
        
        openai_stylometric_sim = calculate_stylometric_similarity(responses['openai'], expected_response)
        gemini_stylometric_sim = calculate_stylometric_similarity(responses['gemini'], expected_response)
        
        results.append({
            'phrase': phrase,
            'expected_response': expected_response,
            'openai_response': responses['openai'],
            'gemini_response': responses['gemini'],
            'category': category,
            'repetition_level': fine_tune_level,
            'openai_levenshtein_similarity': openai_lev_sim,
            'openai_cosine_similarity': openai_cos_sim,
            'gemini_levenshtein_similarity': gemini_lev_sim,
            'gemini_cosine_similarity': gemini_cos_sim,
            'openai_thematic_similarity': openai_thematic_sim,
            'gemini_thematic_similarity': gemini_thematic_sim,
            'openai_stylometric_similarity': openai_stylometric_sim,
            'gemini_stylometric_similarity': gemini_stylometric_sim
        })
    
    return pd.DataFrame(results)

def main():
    output_dir = '/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_2_memorization_threshold/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load synthetic phrases
    phrases_df = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/memorization_data/synthetic_phrases.csv')
    
    # Run the test for each repetition level
    all_results = []
    for repetitions in [10, 50, 100]:  # Simulating exposure levels
        print(f"Testing memorization at repetition level {repetitions}")
        
        # Run memorization test and append results
        results_df = run_memorization_test(fine_tune_level=repetitions, phrases_df=phrases_df)
        all_results.append(results_df)
        
        # Save individual results file
        results_df.to_csv(f'{output_dir}/memorization_results_{repetitions}.csv', index=False)
    
    # Concatenate all results and save summary
    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(f'{output_dir}/memorization_results.csv', index=False)

    # Summarize memorization threshold by category for both models
    summary = all_results_df.groupby(['category', 'repetition_level'])[
        ['openai_levenshtein_similarity', 'openai_cosine_similarity',
         'gemini_levenshtein_similarity', 'gemini_cosine_similarity',
         'openai_thematic_similarity', 'gemini_thematic_similarity',
         'openai_stylometric_similarity', 'gemini_stylometric_similarity']
    ].mean().reset_index()
    
    summary.to_csv(f'{output_dir}/memorization_summary.csv', index=False)
    print("Memorization threshold experiment completed.")

if __name__ == "__main__":
    main()