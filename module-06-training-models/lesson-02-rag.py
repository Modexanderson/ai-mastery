"""
Module 06 -- Lesson 2: RAG (Retrieval-Augmented Generation)
=============================================================
Fine-tuning changes the model itself. But what if you just want
the model to know about YOUR data without retraining?

RAG = give the model relevant context at runtime.

How it works:
  1. You have documents (PDFs, docs, website content, etc.)
  2. Convert them into EMBEDDINGS (numerical vectors)
  3. When user asks a question, find the most SIMILAR documents
  4. Send those documents + the question to the LLM
  5. LLM answers using YOUR data as context

Why RAG is huge for business:
  - No retraining needed (saves time and money)
  - Data stays up-to-date (just add new documents)
  - Works with ANY LLM (Ollama, Claude, GPT)
  - Build chatbots that know your company's docs
  - Build customer support bots, legal assistants, etc.

This lesson builds a complete RAG system from scratch:
  - Document store with embeddings
  - Similarity search (cosine similarity)
  - LLM integration via Ollama
  - All running locally on your machine
"""

import requests
import json
import math
import os
import sys

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"


# ============================================================
# PART 1: Understanding Embeddings
# ============================================================
# An embedding turns text into a vector (list of numbers).
# Similar texts have similar vectors.
# "I love dogs" and "I adore puppies" -> vectors point same direction
# "I love dogs" and "Stock market crash" -> vectors point differently

print("\n" + "=" * 60)
print("  MODULE 06 -- LESSON 2: RAG")
print("  Retrieval-Augmented Generation")
print("  Give any LLM knowledge of YOUR documents")
print("=" * 60)

print("\n  Checking Ollama...")
try:
    requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    print("  Ollama ready!")
except Exception:
    print("  ERROR: Ollama not running! Start with 'ollama serve'")
    sys.exit(1)


# ============================================================
# PART 2: Building Our Own Embeddings
# ============================================================
# Real RAG uses embedding models (sentence-transformers, OpenAI embeddings).
# We'll build a simple TF-IDF-like embedding to understand the concept,
# then use Ollama's built-in embeddings for the real thing.

print("\n" + "=" * 60)
print("  PART 1: Embeddings -- Turning Text into Vectors")
print("=" * 60)


def simple_embedding(text, vocab=None):
    """
    Simple bag-of-words embedding.
    Each word gets a position, text becomes a vector of word counts.
    This is primitive but shows the CONCEPT clearly.
    """
    words = text.lower().split()
    if vocab is None:
        vocab = sorted(set(words))
    vector = [0.0] * len(vocab)
    for word in words:
        if word in vocab:
            idx = vocab.index(word)
            vector[idx] = 1.0
    # Normalize
    magnitude = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / magnitude for v in vector], vocab


def cosine_similarity(vec_a, vec_b):
    """
    Measure how similar two vectors are.
    1.0 = identical direction, 0.0 = completely different
    This is THE standard similarity metric in AI.
    """
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
    mag_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
    return dot / (mag_a * mag_b)


# Demo: show how similar texts have similar embeddings
demo_texts = [
    "python is a programming language",
    "python is used for coding software",
    "the weather is sunny and warm today",
]

# Build shared vocabulary
all_words = sorted(set(w for t in demo_texts for w in t.lower().split()))
embeddings = []
for text in demo_texts:
    vec, _ = simple_embedding(text, all_words)
    embeddings.append(vec)

print(f"\n  Vocabulary: {len(all_words)} unique words")
print(f"\n  Comparing similarity between texts:")
for i in range(len(demo_texts)):
    for j in range(i + 1, len(demo_texts)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"  [{sim:.3f}] '{demo_texts[i][:40]}...'")
        print(f"       vs '{demo_texts[j][:40]}...'")
        print()

print("  Notice: the two Python sentences are more similar (higher score)")
print("  than either is to the weather sentence. That's embeddings working!")


# ============================================================
# PART 3: The Document Store (Knowledge Base)
# ============================================================
print("\n" + "=" * 60)
print("  PART 2: Document Store -- Your Knowledge Base")
print("=" * 60)

# These simulate company documents, FAQs, product info, etc.
# In a real app, you'd load these from PDFs, databases, websites.
DOCUMENTS = [
    {
        "id": 1,
        "title": "Return Policy",
        "content": "Our return policy allows customers to return any product within 30 days of purchase for a full refund. Items must be in original packaging and unused condition. Digital products are non-refundable after download. To initiate a return, contact support with your order number."
    },
    {
        "id": 2,
        "title": "Shipping Information",
        "content": "We offer free standard shipping on orders over 50 dollars. Standard shipping takes 5 to 7 business days. Express shipping is available for 15 dollars and delivers in 2 to 3 business days. International shipping costs vary by location and takes 10 to 15 business days."
    },
    {
        "id": 3,
        "title": "Product: AI Writing Assistant",
        "content": "Our AI Writing Assistant helps you write better content faster. It supports blog posts, emails, social media, and marketing copy. Pricing starts at 19 dollars per month for the basic plan with 50000 words. The pro plan at 49 dollars per month includes unlimited words and priority support."
    },
    {
        "id": 4,
        "title": "Product: Smart Security Camera",
        "content": "Our Smart Security Camera features real-time person detection using AI, night vision, two-way audio, and cloud recording. It works with WiFi and stores 30 days of footage. The camera costs 99 dollars with an optional cloud plan at 5 dollars per month."
    },
    {
        "id": 5,
        "title": "Technical Support",
        "content": "For technical issues, try restarting the device first. If the problem persists, check our troubleshooting guide at support dot example dot com. You can reach our support team via live chat Monday to Friday 9am to 6pm, or email support at example dot com for 24 hour response."
    },
    {
        "id": 6,
        "title": "Company Information",
        "content": "Founded in 2023, we are a technology company focused on AI-powered products for small businesses. Our team of 25 engineers builds tools that automate repetitive tasks. We are based in Austin, Texas and serve customers in 40 countries."
    },
    {
        "id": 7,
        "title": "Privacy Policy",
        "content": "We take privacy seriously. Customer data is encrypted at rest and in transit. We never sell personal information to third parties. Users can request data deletion at any time. We comply with GDPR and CCPA regulations."
    },
    {
        "id": 8,
        "title": "Pricing and Plans",
        "content": "We offer three plans: Starter at 0 dollars for individuals with basic features, Professional at 29 dollars per month for small teams with advanced features and priority support, and Enterprise with custom pricing for large organizations with dedicated support and SLA guarantees."
    },
]

print(f"\n  Loaded {len(DOCUMENTS)} documents into knowledge base:")
for doc in DOCUMENTS:
    print(f"    {doc['id']}. {doc['title']} ({len(doc['content'])} chars)")


# ============================================================
# PART 4: Building the RAG Retriever
# ============================================================
print("\n" + "=" * 60)
print("  PART 3: RAG Retriever -- Finding Relevant Documents")
print("=" * 60)


class SimpleRAG:
    """
    A complete RAG system:
    1. Index documents (create embeddings)
    2. Retrieve relevant docs for a query
    3. Generate answer using LLM + retrieved context
    """

    def __init__(self, documents):
        self.documents = documents
        self.vocab = self._build_vocab()
        self.doc_embeddings = self._embed_documents()
        print(f"\n  RAG system initialized!")
        print(f"  Documents: {len(documents)}")
        print(f"  Vocabulary: {len(self.vocab)} words")

    def _build_vocab(self):
        """Build vocabulary from all documents."""
        words = set()
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']}"
            for word in text.lower().split():
                words.add(word)
        return sorted(words)

    def _embed_documents(self):
        """Create embeddings for all documents."""
        embeddings = []
        for doc in self.documents:
            text = f"{doc['title']} {doc['content']}"
            vec, _ = simple_embedding(text, self.vocab)
            embeddings.append(vec)
        return embeddings

    def retrieve(self, query, top_k=3):
        """Find the most relevant documents for a query."""
        query_vec, _ = simple_embedding(query, self.vocab)

        # Calculate similarity to every document
        scores = []
        for i, doc_vec in enumerate(self.doc_embeddings):
            sim = cosine_similarity(query_vec, doc_vec)
            scores.append((sim, i))

        # Sort by similarity (highest first)
        scores.sort(reverse=True)

        # Return top-k documents
        results = []
        for sim, idx in scores[:top_k]:
            results.append({
                "document": self.documents[idx],
                "similarity": sim,
            })
        return results

    def generate_answer(self, query):
        """Full RAG pipeline: retrieve + generate."""
        # Step 1: Retrieve relevant documents
        results = self.retrieve(query, top_k=3)

        print(f"\n  Retrieved {len(results)} relevant documents:")
        for r in results:
            print(f"    [{r['similarity']:.3f}] {r['document']['title']}")

        # Step 2: Build context from retrieved documents
        context = "\n\n".join(
            f"Document: {r['document']['title']}\n{r['document']['content']}"
            for r in results
        )

        # Step 3: Send to LLM with context
        prompt = f"""Use ONLY the following documents to answer the question.
If the answer is not in the documents, say "I don't have that information."

DOCUMENTS:
{context}

QUESTION: {query}

ANSWER (be concise, 1-3 sentences):"""

        print(f"\n  Generating answer...")
        sys.stdout.write("  Answer: ")
        sys.stdout.flush()

        try:
            response = requests.post(f"{OLLAMA_URL}/api/chat", json={
                "model": MODEL,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"num_ctx": 2048, "num_predict": 150, "num_gpu": 99},
            }, stream=True, timeout=300)

            answer = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    answer += token
                    sys.stdout.write(token)
                    sys.stdout.flush()
            print()
            return answer
        except Exception as e:
            print(f"\n  ERROR: {e}")
            return "Error generating answer."


# Initialize RAG
rag = SimpleRAG(DOCUMENTS)


# ============================================================
# PART 5: Demo Queries
# ============================================================
print("\n" + "=" * 60)
print("  PART 4: Demo -- Watch RAG Answer Questions")
print("=" * 60)

demo_questions = [
    "What is your return policy?",
    "How much does the security camera cost?",
    "What are the pricing plans?",
]

for question in demo_questions:
    print(f"\n  Question: {question}")
    print("  " + "-" * 40)
    rag.generate_answer(question)
    print()


# ============================================================
# PART 6: Interactive Mode
# ============================================================
print("\n" + "=" * 60)
print("  INTERACTIVE MODE")
print("  Ask any question about the company/products!")
print("  The RAG system finds relevant docs and answers.")
print("  Type 'docs' to see all documents, 'quit' to exit")
print("=" * 60)

while True:
    try:
        query = input("\n  Your question: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not query:
        continue
    if query.lower() in ["quit", "exit", "q"]:
        break
    if query.lower() == "docs":
        for doc in DOCUMENTS:
            print(f"    {doc['id']}. {doc['title']}")
        continue

    rag.generate_answer(query)

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS:")
print("  1. RAG = Retrieve relevant docs + Generate answer with LLM")
print("  2. Embeddings turn text into vectors for similarity search")
print("  3. No retraining needed -- just add/update documents")
print("  4. This is how ChatGPT plugins, Perplexity, and company")
print("     chatbots work")
print("  5. For production: use sentence-transformers or OpenAI")
print("     embeddings instead of bag-of-words")
print("  6. Store embeddings in vector databases (Pinecone, ChromaDB)")
print("=" * 60)
