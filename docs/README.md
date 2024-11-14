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