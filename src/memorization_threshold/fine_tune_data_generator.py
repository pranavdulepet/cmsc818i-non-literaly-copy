# src/fine_tune_data_generator.py

import pandas as pd
import os

def generate_fine_tuning_data(input_file, output_dir, repetitions_list):
    # Load synthetic phrases
    phrases_df = pd.read_csv(input_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for repetitions in repetitions_list:
        # Generate repeated phrases for each repetition level
        repeated_phrases = []
        for _, row in phrases_df.iterrows():
            repeated_phrases.extend([row.to_dict()] * repetitions)
        
        # Save each repetition level to a separate file
        output_file = os.path.join(output_dir, f'synthetic_phrases_repeated_{repetitions}.csv')
        pd.DataFrame(repeated_phrases).to_csv(output_file, index=False)
        print(f"Generated fine-tuning data with {repetitions} repetitions at {output_file}")

if __name__ == "__main__":
    generate_fine_tuning_data(
        input_file='/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/memorization_data/synthetic_phrases.csv',
        output_dir='/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/data/memorization_data/fine_tune_data',
        repetitions_list=[10, 50, 100]
    )