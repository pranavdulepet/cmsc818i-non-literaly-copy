# src/memorization_threshold.py

import pandas as pd
from api_interface import get_openai_response
from Levenshtein import distance as levenshtein_distance
from sentence_transformers import SentenceTransformer, util
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(expected, response):
    # Levenshtein similarity (normalized)
    lev_sim = 1 - (levenshtein_distance(expected, response) / max(len(expected), len(response)))
    
    # Cosine similarity
    embeddings = model.encode([expected, response])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return lev_sim, cos_sim

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
        
        # Get model response using OpenAI API
        model_response = get_openai_response(prompt)
        
        # Calculate similarity scores
        lev_sim, cos_sim = calculate_similarity(expected_response, model_response)
        
        results.append({
            'phrase': phrase,
            'expected_response': expected_response,
            'model_response': model_response,
            'category': category,
            'repetition_level': fine_tune_level,
            'levenshtein_similarity': lev_sim,
            'cosine_similarity': cos_sim
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

    # Summarize memorization threshold by category
    summary = all_results_df.groupby(['category', 'repetition_level'])[['levenshtein_similarity', 'cosine_similarity']].mean().reset_index()
    summary.to_csv(f'{output_dir}/memorization_summary.csv', index=False)
    print("Memorization threshold experiment completed.")

if __name__ == "__main__":
    main()