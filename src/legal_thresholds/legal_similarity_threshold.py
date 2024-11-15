import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_legal_similarity(response, reference_text):
    # Compute similarity
    embeddings = model.encode([response, reference_text])
    cos_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return cos_sim

def legal_threshold_experiment():
    data = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/non_literal_copying_data/copyrighted_texts.csv')
    responses = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv')
    
    legal_threshold_results = []
    for _, response_row in responses.iterrows():
        for _, data_row in data.iterrows():
            similarity_score = calculate_legal_similarity(response_row['model_response'], data_row['text_excerpt'])
            legal_threshold_results.append({
                'prompt': response_row['prompt'],
                'model_response': response_row['model_response'],
                'reference_text': data_row['text_excerpt'],
                'similarity_score': similarity_score,
                'meets_threshold': similarity_score > 0.7  # Example threshold
            })
    
    pd.DataFrame(legal_threshold_results).to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_6_legal_thresholds/results/legal_similarity_results.csv', index=False)
