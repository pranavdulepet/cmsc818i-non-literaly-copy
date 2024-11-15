import pandas as pd
from sentence_transformers import SentenceTransformer, util
from Levenshtein import distance as levenshtein_distance

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_levenshtein_similarity(text1, text2):
    """Calculate normalized Levenshtein similarity between two texts."""
    lev_sim = 1 - (levenshtein_distance(text1, text2) / max(len(text1), len(text2), 1))
    return lev_sim

def calculate_cosine_similarity(response, reference_text):
    """Calculate cosine similarity using sentence embeddings."""
    embeddings = model.encode([response, reference_text])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cos_sim

def non_literal_similarity_experiment():
    # Load data files
    copyrighted_texts = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/non_literal_copying_data/copyrighted_texts.csv')
    responses = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv')
    
    results = []
    
    for _, row in responses.iterrows():
        prompt = row['prompt']
        response = row['model_response']
        category = row['category']
        
        for _, ref_row in copyrighted_texts.iterrows():
            reference_text = ref_row['text_excerpt']
            ref_title = ref_row['title']
            
            # Calculate Levenshtein and cosine similarity
            levenshtein_similarity = calculate_levenshtein_similarity(response, reference_text)
            cosine_similarity = calculate_cosine_similarity(response, reference_text)
            
            # Record results
            results.append({
                'prompt': prompt,
                'model_response': response,
                'category': category,
                'reference_title': ref_title,
                'levenshtein_similarity': levenshtein_similarity,
                'cosine_similarity': cosine_similarity
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_6_legal_thresholds/results/results/non_literal_similarity_results.csv', index=False)
    print("Non-literal similarity comparison completed.")

if __name__ == "__main__":
    non_literal_similarity_experiment()
