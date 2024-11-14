# src/rephrase_prompt.py

import random

def rephrase_prompt(prompt, iteration):
    """Generates a rephrased version of the input prompt based on iteration step."""
    # Simple rephrasing rules for demonstration
    rephrases = [
        lambda p: f"Can you explain {p}?",
        lambda p: f"In your opinion, {p}",
        lambda p: f"Please describe {p}",
        lambda p: f"What can you say about {p}?",
        lambda p: f"Tell me about {p}"
    ]
    return rephrases[iteration % len(rephrases)](prompt)