import pandas as pd
from src.api_interface import get_openai_response, get_gemini_response, get_dual_responses
from sentence_transformers import SentenceTransformer, util
from src.semantic_drift.rephrase_prompt import rephrase_prompt
from src.stylometric_analysis import calculate_stylometric_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_cosine_similarity(response1, response2):
    """Calculates cosine similarity between two responses."""
    embeddings = model.encode([response1, response2])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cos_sim

def calculate_thematic_similarity(response, reference):
    """Calculates thematic similarity between two responses."""
    embeddings = model.encode([response, reference])
    return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

def run_drift_experiment(initial_prompts, num_rephrases=5):
    results = []
    
    for _, row in initial_prompts.iterrows():
        original_prompt = row['prompt']
        category = row['category']
        
        # Get initial responses from both models
        prompt = original_prompt
        responses = get_dual_responses(prompt)
        previous_responses = responses
        
        for i in range(1, num_rephrases + 1):
            # Generate rephrased prompt
            prompt = rephrase_prompt(original_prompt, i)
            
            # Get model responses for rephrased prompt
            model_responses = get_dual_responses(prompt)
            
            # Calculate cosine similarity for both models
            openai_cosine_similarity = calculate_cosine_similarity(
                previous_responses['openai'], 
                model_responses['openai']
            )
            gemini_cosine_similarity = calculate_cosine_similarity(
                previous_responses['gemini'], 
                model_responses['gemini']
            )
            
            # Calculate thematic similarity for non-literal drift
            openai_thematic_similarity = calculate_thematic_similarity(
                model_responses['openai'], 
                original_prompt
            )
            gemini_thematic_similarity = calculate_thematic_similarity(
                model_responses['gemini'], 
                original_prompt
            )

            # Calculate stylometric similarity for non-literal style drift
            openai_stylometric_similarity = calculate_stylometric_similarity(
                previous_responses['openai'], 
                model_responses['openai']
            )
            gemini_stylometric_similarity = calculate_stylometric_similarity(
                previous_responses['gemini'], 
                model_responses['gemini']
            )
            
            results.append({
                'original_prompt': original_prompt,
                'category': category,
                'rephrase_step': i,
                'rephrased_prompt': prompt,
                'openai_response': model_responses['openai'],
                'gemini_response': model_responses['gemini'],
                'openai_cosine_similarity': openai_cosine_similarity,
                'gemini_cosine_similarity': gemini_cosine_similarity,
                'openai_thematic_similarity': openai_thematic_similarity,
                'gemini_thematic_similarity': gemini_thematic_similarity,
                'openai_stylometric_similarity': openai_stylometric_similarity,
                'gemini_stylometric_similarity': gemini_stylometric_similarity
            })
            
            # Update previous responses for the next iteration
            previous_responses = model_responses
    
    return pd.DataFrame(results)

def main():
    # Load initial prompts
    initial_prompts = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/data_drift/initial_prompts.csv')
    
    # Run the drift experiment
    results_df = run_drift_experiment(initial_prompts)
    
    # Save detailed drift results
    results_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_3_semantic_drift/results/drift_results.csv', index=False)
    
    # Summarize drift by averaging similarity scores for each category and step
    summary_df = results_df.groupby(['category', 'rephrase_step'])[
        ['openai_cosine_similarity', 'gemini_cosine_similarity', 'openai_thematic_similarity', 
         'gemini_thematic_similarity', 'openai_stylometric_similarity', 'gemini_stylometric_similarity']
    ].mean().reset_index()
    
    summary_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_3_semantic_drift/results/drift_summary.csv', index=False)
    print("Semantic drift experiment completed.")

if __name__ == "__main__":
    main()