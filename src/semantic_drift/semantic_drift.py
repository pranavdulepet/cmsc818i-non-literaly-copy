# src/semantic_drift.py

import pandas as pd
from api_interface import get_openai_response
from sentence_transformers import SentenceTransformer, util
from rephrase_prompt import rephrase_prompt

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(response1, response2):
    """Calculates cosine similarity between two responses."""
    embeddings = model.encode([response1, response2])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cos_sim

def run_drift_experiment(initial_prompts, num_rephrases=5):
    results = []
    
    for _, row in initial_prompts.iterrows():
        original_prompt = row['prompt']
        category = row['category']
        
        # Initialize previous response to the first model response for the original prompt
        prompt = original_prompt
        previous_response = get_openai_response(prompt)
        
        for i in range(1, num_rephrases + 1):
            # Generate rephrased prompt
            prompt = rephrase_prompt(original_prompt, i)
            
            # Get model response for rephrased prompt
            model_response = get_openai_response(prompt)
            
            # Calculate cosine similarity with the previous response
            cosine_similarity = calculate_cosine_similarity(previous_response, model_response)
            
            results.append({
                'original_prompt': original_prompt,
                'category': category,
                'rephrase_step': i,
                'rephrased_prompt': prompt,
                'model_response': model_response,
                'cosine_similarity': cosine_similarity
            })
            
            # Update previous response for the next comparison
            previous_response = model_response
    
    return pd.DataFrame(results)

def main():
    # Load initial prompts
    initial_prompts = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/data_drift/initial_prompts.csv')
    
    # Run the drift experiment
    results_df = run_drift_experiment(initial_prompts)
    
    # Save detailed drift results
    results_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_3_semantic_drift/results/drift_results.csv', index=False)
    
    # Summarize drift by averaging similarity scores for each category and step
    summary_df = results_df.groupby(['category', 'rephrase_step'])['cosine_similarity'].mean().reset_index()
    summary_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_3_semantic_drift/results/drift_summary.csv', index=False)
    print("Semantic drift experiment completed.")

if __name__ == "__main__":
    main()
