import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from textstat import textstat

model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_tfidf_similarity(text1, text2):
    """Calculate TF-IDF cosine similarity between two texts."""
    vectorizer = TfidfVectorizer().fit([text1, text2])
    vectors = vectorizer.transform([text1, text2])
    cosine_similarity = (vectors * vectors.T).toarray()[0, 1]
    return cosine_similarity

def calculate_stylometric_features(text):
    """Calculate a set of stylometric features for a given text."""
    features = {
        'lexical_density': textstat.lexicon_count(text) / max(1, len(text.split())),  # words per sentence
        'average_sentence_length': textstat.avg_sentence_length(text),
        'syllable_count_per_word': textstat.avg_syllables_per_word(text),
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
    }
    return features

def calculate_stylometric_similarity(text1, text2):
    """Calculate stylometric similarity between two texts."""
    features1 = calculate_stylometric_features(text1)
    features2 = calculate_stylometric_features(text2)
    score = sum(1 - abs(features1[key] - features2[key]) / max(features1[key], features2[key], 1) 
                for key in features1) / len(features1)
    return score

def stylometric_analysis():
    # Load data files
    copyrighted_texts = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/non_literal_copying_data/copyrighted_texts.csv')
    responses = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv')
    
    results = []
    
    for _, row in responses.iterrows():
        prompt = row['prompt']
        response = row['model_response']
        category = row['category']
        
        # Compare with each copyrighted text
        for _, ref_row in copyrighted_texts.iterrows():
            reference_text = ref_row['text_excerpt']
            ref_title = ref_row['title']
            
            # TF-IDF similarity for thematic similarity
            tfidf_similarity = calculate_tfidf_similarity(response, reference_text)
            
            # Stylometric similarity for style comparison
            response_features = calculate_stylometric_features(response)
            reference_features = calculate_stylometric_features(reference_text)
            stylometric_similarity = calculate_stylometric_similarity(response_features, reference_features)
            
            # Cosine similarity for an additional similarity metric
            embeddings = model.encode([response, reference_text])
            cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
            
            results.append({
                'prompt': prompt,
                'model_response': response,
                'category': category,
                'reference_title': ref_title,
                'tfidf_similarity': tfidf_similarity,
                'stylometric_similarity': stylometric_similarity,
                'cosine_similarity': cosine_similarity
            })
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_5_stylometry/results/stylometric_analysis.csv', index=False)
    print("Stylometric and thematic analysis completed.")

if __name__ == "__main__":
    stylometric_analysis()