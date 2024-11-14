import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

results_df = pd.read_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/trigger_analysis_results.csv')

def is_interpretive(response):
    keywords = ['could', 'might', 'perhaps', 'possibly', 'seems', 'appears']
    return any(keyword in response.lower() for keyword in keywords)

# Label interpretive responses
results_df['interpretive'] = results_df.apply(
    lambda row: is_interpretive(row['model_response']) if row['category'] in ['Rare Phrases', 'IP-Like'] else False,
    axis=1
)

interpretive_analysis = results_df[results_df['interpretive']]
interpretive_analysis.to_csv('/Users/macbookair/Desktop/desktop-subfolder/umd/cmsc818i/cmsc818i-selective-forgetting/experiments/experiment_1_trigger_analysis/results/interpretive_analysis.csv', index=False)

interpretive_proportion = results_df.groupby('category')['interpretive'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=interpretive_proportion, x='category', y='interpretive')
plt.title('Proportion of Interpretive Responses by Category')
plt.ylabel('Proportion of Interpretive Responses')
plt.xlabel('Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()