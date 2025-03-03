import os
import random
import re
import string
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

# Set a random seed for reproducibility
random.seed(42)

# Create a directory if it doesn't already exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def compute_similarity_metrics(vec1, vec2):
    cos_sim = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    euc_dist = euclidean(vec1, vec2)
    pearson_corr, _ = pearsonr(vec1, vec2)
    
    return cos_sim, euc_dist, pearson_corr

def prompt_squad(question, context):
    question = question
    context = context
    line1 = "Answer the question with a word or phrase based on the context below:"
    line2 = "\nQuestion: {}".format(question)
    context_label = "\nContext: "
    offset = len(line1 + line2 + context_label)
    prompt = "Answer the question with a word or phrase based on the context below:"
    prompt += "\nQuestion: {}".format(question)
    prompt += "\nContext: {}".format(context)
    prompt += "\nResponse in the following format without any other information:"
    prompt += "\n>reason: {reason for the answer here}"
    prompt += "\n>answer: {answer here}"
    return prompt