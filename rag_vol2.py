#%% Imports

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import requests
from sentence_transformers import CrossEncoder

#%% Getting Text Data

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

text = load_text(path = r"file path")

#%% Langchaing Chunking and Embedding

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        ""
    ]
)

chunks = splitter.split_text(text)

embed_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

embeddings = embed_model.encode(
    chunks,
    normalize_embeddings=True
).astype("float32")

#%% Faiss Indexing

def build_index(embeddings):
    dim = embeddings.shape[1]
    
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 40
    index.hnsw.efSearch = 32
    
    index.add(embeddings)
    
    return index
    
index = build_index(embeddings)

#%% Example Query with LLM

reranker = CrossEncoder("BAAI/bge-reranker-base")

def ask_llm(query):
    query_embedding = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")
    
    D, I = index.search(query_embedding, k=10)
    
    candidates = [chunks[i] for i in I[0]]
    
    pairs = [(query, chunk) for chunk in candidates]
    
    scores = reranker.predict(pairs)
    
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    
    top_chunks = [c[0] for c in ranked[:3]]
    
    context = "\n\n".join(top_chunks)
    
    prompt = f"""
    Answer the question using ONLY the provided context.
    If the answer cannot be found in the context, say that you don't know.
    
    Context:
    {context}
    
    Question:
    {query}
    
    Answer:
    """
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen3:4b",
                "prompt": prompt,
                "stream": False
            }
        )
        response.raise_for_status()
    
        print(response.json()["response"])

    except Exception as e:
        print(f"Error: {e}")
    
query = "Which educations does this person have?"
ask_llm(query)