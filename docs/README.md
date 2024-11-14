# cmsc818i-selective-forgetting

Title: 
Quantifying and Explaining Memorization in LLM Responses: Identifying Security and Privacy Risks in AI-Generated Outputs

Objective:
This project aims to investigate and quantify memorization in large language models (LLMs) to better understand when and why these models tend to "memorize" and potentially regurgitate training data. By exploring this phenomenon, the project will assess privacy and security risks associated with memorization, particularly where sensitive information is involved. The project will also focus on visualizing decision pathways to make the memorization process more interpretable and develop metrics to quantify the likelihood and impact of memorization in LLMs.

Background & Motivation:
As LLMs grow in scale and application, they increasingly interact with diverse and potentially sensitive datasets. Instances where these models output training data verbatim or with high similarity raise concerns around privacy, data security, and ethical AI use, especially for applications that must comply with privacy regulations like GDPR. Previous studies have focused on unlearning and selective forgetting of sensitive information. However, few have explored the mechanisms behind memorization, quantified its prevalence, or visualized the internal processes that lead to memorized responses.

The insights gained from this study could enable safer and more responsible deployment of LLMs by identifying and mitigating memorization-related risks. By building tools to detect, quantify, and explain memorization, this research aims to contribute to transparency in AI model behavior and enhance the security and privacy safeguards in LLM applications.

Research Questions:
1. Detection and Analysis of Memorization: How can we detect instances of memorization in LLM outputs? What types of inputs (triggers) lead LLMs to produce memorized content?
2. Quantifying Memorization: Can we develop a metric to measure the likelihood of memorization in LLMs? How can we determine the extent to which an output is memorized versus generated?
3. Privacy and Security Implications: What are the privacy risks associated with memorized responses? Under what conditions does memorization create security vulnerabilities?
4. Interpretability of Memorization Pathways: How can we visualize the decision pathways that lead LLMs to regurgitate training data? What interpretability tools can best illustrate the difference between memorization and generalization in responses?

Experiments:
### 1. Trigger Analysis

**Objective**:  
To evaluate the model's response patterns and similarity to expected answers across different categories of prompts, including sensitive information, common knowledge, rare phrases, and IP-like terms. This experiment assesses how closely the model matches or generalizes content, both for original and rephrased prompts.

**Methodology**:  
- Original and rephrased prompts were analyzed to determine Levenshtein and cosine similarity scores between the expected and model-generated responses.
- Categories examined: Sensitive Info, Common Knowledge, Rare Phrases, and IP-Like.
- Interpretive responses were identified and quantified, especially for ambiguous prompts.

**Key Findings**:
- Sensitive prompts consistently triggered refusal responses, demonstrating robust privacy safeguards.
- Common knowledge prompts resulted in high similarity scores, indicating factual recall.
- Rare phrases and IP-like terms prompted interpretive responses, with moderate similarity scores and variability upon rephrasing.

**Conclusion**:  
The trigger analysis highlights that the model effectively avoids verbatim memorization of sensitive information while recalling common knowledge and providing interpretive responses for ambiguous terms.

---

### 2. Memorization Threshold Experiment

**Objective**:  
To determine at what point repeated exposure to specific phrases or information in the training data leads the model to memorize and reproduce these phrases exactly. This experiment aims to identify a "memorization threshold" based on frequency.

**Methodology**:
- Create synthetic phrases or canary phrases with varying repetition frequencies in a controlled dataset used to fine-tune the model.
- Measure the frequency at which the model begins to output these phrases verbatim in response to related prompts.
- Compare across categories to determine if the memorization threshold differs between sensitive, common, and unique phrases.

**Expected Outcome**:  
This experiment should reveal whether certain categories of information have lower or higher memorization thresholds. For instance, sensitive information may require more frequent exposure to be memorized, suggesting that the model is less prone to memorize sensitive data unless it is highly prevalent.

---

### 3. Semantic Drift with Rephrased Prompts

**Objective**:  
To explore if rephrasing prompts progressively leads the model away from the original meaning, especially for ambiguous and interpretive content. This experiment assesses the stability of the model’s interpretive responses when prompts are rephrased repeatedly.

**Methodology**:
- Start with a set of original prompts and iteratively rephrase them, creating a “chain” of rephrased prompts.
- Measure semantic similarity at each rephrasing step using cosine similarity between responses.
- Track if and when the model’s responses begin to deviate significantly from the original expected response, particularly for Rare Phrases and IP-Like categories.

**Expected Outcome**:  
The experiment is expected to show that interpretive content (e.g., rare phrases, IP-like terms) is more prone to semantic drift over rephrasing iterations, while factual content (e.g., common knowledge) remains relatively stable.

---

### 4. Privacy Sensitivity Test

**Objective**:  
To evaluate the model’s sensitivity to privacy-related terms by testing prompts that suggest confidentiality. This experiment examines if the model treats certain keywords (e.g., "confidential," "private") as signals to avoid responding with memorized content.

**Methodology**:
- Design prompts that vary in sensitivity, using keywords like "confidential," "private," and "classified."
- Include a mix of sensitive and non-sensitive prompts, both with and without these privacy-related keywords.
- Analyze refusal rates and similarity scores to see if privacy keywords lead to more refusal responses or lower similarity.

**Expected Outcome**:  
This experiment should reveal whether the model's privacy mechanism is influenced by specific keywords, enhancing its ability to avoid potential privacy violations.

---

### 5. Interpretive and Speculative Response Analysis

**Objective**:  
To analyze and quantify the frequency and type of speculative language (e.g., “might,” “could”) in responses to ambiguous prompts. This experiment provides insight into the model's interpretive behavior and its use of speculative language when exact answers are unavailable.

**Methodology**:
- Identify interpretive prompts (e.g., rare phrases, IP-like terms) and track the use of speculative keywords in the responses.
- Measure the proportion of responses containing speculative language and compare across categories.
- Analyze if certain prompt structures or content types trigger speculative responses more frequently.

**Expected Outcome**:  
We expect the model to use speculative language predominantly for rare and IP-like terms, indicating a fallback to interpretive behavior when lacking concrete information.

---

### 6. Retention vs. Forgetting Experiment (Optional)

**Objective**:  
To explore if the model can “forget” specific information when prompted to do so, examining its retention capabilities in light of selective unlearning requests.

**Methodology**:
- Introduce specific phrases in a fine-tuning dataset, followed by prompts asking the model to “forget” these phrases by re-finetuning with counter-prompts (e.g., "Forget that X is true").
- Test the model’s responses to determine if it still recalls the initial phrase, and measure similarity scores to gauge retention.

**Expected Outcome**:  
This experiment would provide insight into the model’s ability to retain or discard information based on additional fine-tuning, offering potential applications in controlled unlearning or data purging.