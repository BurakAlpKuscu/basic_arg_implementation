#%% text

with open(r"file path", "r", encoding="utf-8") as f:
    text = f.read()
    
#%% Size based chunking

def chunk_text_size(text, chunk_size=300):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

size_chunks = chunk_text_size(text, chunk_size = 400)

#%% Overlap and sentence chunking

import nltk
# nltk.download('punkt_tab')

def chunk_text_overlap(text,
               max_chunk_size=500,
               overlap_sentences=1):

    sentences = nltk.sent_tokenize(text)

    chunks = []
    current = []

    current_length = 0

    for sentence in sentences:

        if current_length + len(sentence) <= max_chunk_size:
            current.append(sentence)
            current_length += len(sentence)

        else:
            chunks.append(" ".join(current))

            current = current[-overlap_sentences:]
            current_length = sum(len(s) for s in current)

            current.append(sentence)
            current_length += len(sentence)

    if current:
        chunks.append(" ".join(current))

    return chunks

overlap_chunks = chunk_text_overlap(text, max_chunk_size=600, overlap_sentences=1)

#%% Semantic Chunking

from sentence_transformers import SentenceTransformer
import numpy as np
import nltk

# nltk.download("punkt")

model = SentenceTransformer("all-MiniLM-L6-v2")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def split_sentences(text):
    return nltk.sent_tokenize(text)

def semantic_chunk(text):
    sentences = split_sentences(text)

    embeddings = model.encode(sentences)

    chunks = []
    current_embeddings = [embeddings[0]]
    current_chunk = [sentences[0]]
    
    test_embeddings = [embeddings[0]]
    centroid_sims = []
    
    for i in range(1, len(sentences)):
        center = np.mean(test_embeddings, axis=0)
    
        sim = cosine_similarity(center, embeddings[i])
    
        centroid_sims.append(sim)
    
        test_embeddings.append(embeddings[i])
    
    threshold = np.percentile(centroid_sims, 20)
    
    print("Threshold: ", threshold)
    
    for i in range(1, len(sentences)):
        chunk_center = np.mean(current_embeddings, axis=0)
    
        sim = cosine_similarity(
            chunk_center,
            embeddings[i]
        )
    
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
    
            current_chunk = [sentences[i]]
            current_embeddings = [embeddings[i]]
        else:
            current_chunk.append(sentences[i])
            current_embeddings.append(embeddings[i])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

semantic_chunks = semantic_chunk(text)
