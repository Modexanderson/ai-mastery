"""
Module 06 -- Lesson 1: Fine-Tuning a Language Model
=====================================================
So far we've USED pre-trained models (Ollama, Whisper, YOLO).
Now we learn to CUSTOMIZE them.

What is fine-tuning?
  A pre-trained model already knows language/vision/etc.
  Fine-tuning = teaching it YOUR specific style, domain, or task.

Analogy:
  Pre-trained model = a college graduate (knows general stuff)
  Fine-tuning       = job training (now they're specialized)

Types of training:
  1. FROM SCRATCH   -- train on raw data (needs millions of examples, huge GPU)
  2. FINE-TUNING    -- take a trained model, adjust it on small data (100-1000 examples)
  3. LoRA/QLoRA     -- fine-tune only a TINY part of the model (works on consumer GPUs!)
  4. RAG            -- don't retrain at all, just give context at runtime

This lesson: Fine-tune a small text classifier using PyTorch.
We'll train a model to classify movie reviews as positive or negative.
This teaches the CORE concepts that apply to fine-tuning any model.

Hardware: Your RTX 4070 (8GB VRAM) can fine-tune models up to ~7B parameters!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# ============================================================
# PART 1: Understanding Tokenization
# ============================================================

print("\n" + "=" * 60)
print("  MODULE 06 -- LESSON 1: FINE-TUNING A MODEL")
print("  Training a Sentiment Classifier with PyTorch")
print("=" * 60)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Device: {device}", end="")
if device.type == "cuda":
    print(f" ({torch.cuda.get_device_name(0)})")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print(" (CPU -- still works, just slower)")

# -- Training Data --
TRAIN_DATA = [
    ("this movie was absolutely amazing and wonderful", 1),
    ("great film with brilliant acting and story", 1),
    ("loved every minute of this beautiful movie", 1),
    ("excellent performances and stunning visuals", 1),
    ("one of the best films i have ever seen", 1),
    ("fantastic story with incredible acting", 1),
    ("truly a masterpiece of modern cinema", 1),
    ("heartwarming and deeply moving experience", 1),
    ("outstanding direction and superb screenplay", 1),
    ("a delightful and entertaining film", 1),
    ("wonderful characters and engaging plot", 1),
    ("brilliant cinematography and perfect pacing", 1),
    ("this film exceeded all my expectations", 1),
    ("an inspiring and uplifting story", 1),
    ("beautifully crafted with amazing performances", 1),
    ("a joy to watch from start to finish", 1),
    ("superb acting and gripping storyline", 1),
    ("one of the greatest movies ever made", 1),
    ("absolutely loved this incredible film", 1),
    ("perfect blend of humor and drama", 1),
    ("this movie was terrible and boring", 0),
    ("awful acting and a weak storyline", 0),
    ("waste of time and money watching this", 0),
    ("horrible film with no redeeming qualities", 0),
    ("one of the worst movies ever made", 0),
    ("dull and predictable from start to finish", 0),
    ("terrible direction and poor screenplay", 0),
    ("completely unwatchable and painfully slow", 0),
    ("disappointing and frustrating experience", 0),
    ("a disaster of a film", 0),
    ("boring characters and meaningless plot", 0),
    ("poorly made with awful special effects", 0),
    ("this film was a complete letdown", 0),
    ("dreadful acting and nonsensical story", 0),
    ("an absolute waste of talent", 0),
    ("unbearable to sit through this mess", 0),
    ("so bad it made me want to leave", 0),
    ("forgettable and utterly mediocre", 0),
    ("painfully unfunny and cringe worthy", 0),
    ("worst movie of the year by far", 0),
]

TEST_DATA = [
    ("a truly wonderful and amazing experience", 1),
    ("brilliant movie with perfect acting", 1),
    ("absolutely terrible waste of time", 0),
    ("horrible and boring film", 0),
    ("loved this fantastic masterpiece", 1),
    ("dull awful disaster of a movie", 0),
    ("incredible story beautifully told", 1),
    ("worst film with dreadful acting", 0),
]


# ============================================================
# PART 2: Building a Vocabulary (Tokenizer)
# ============================================================
print("\n" + "=" * 60)
print("  PART 1: Tokenization -- Turning Words into Numbers")
print("=" * 60)


def build_vocab(data):
    """Build a word-to-index mapping from training data."""
    words = set()
    for text, _ in data:
        for word in text.lower().split():
            words.add(word)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, word in enumerate(sorted(words), start=2):
        vocab[word] = i
    return vocab


vocab = build_vocab(TRAIN_DATA)
print(f"\n  Vocabulary size: {len(vocab)} words")
print(f"  Sample mappings:")
for w in ["amazing", "terrible", "movie", "film", "boring"]:
    print(f"    '{w}' -> {vocab.get(w, vocab['<UNK>'])}")


def text_to_tensor(text, vocab, max_len=15):
    """Convert text to a fixed-length tensor of word indices."""
    words = text.lower().split()
    indices = [vocab.get(w, vocab["<UNK>"]) for w in words]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)


demo_text = "this movie was amazing"
demo_tensor = text_to_tensor(demo_text, vocab)
print(f"\n  Example: '{demo_text}'")
print(f"  Tensor:  {demo_tensor.tolist()}")


# ============================================================
# PART 3: The Model Architecture
# ============================================================
print("\n" + "=" * 60)
print("  PART 2: Model Architecture -- Embedding + Classifier")
print("=" * 60)


class SentimentClassifier(nn.Module):
    """
    Architecture:
      1. Embedding: word indices -> 64-dim vectors
      2. Average: all word vectors -> one sentence vector
      3. Hidden: 64 -> 32 neurons (learn patterns)
      4. Output: 32 -> 1 (positive or negative)

    Same CONCEPT as BERT/GPT fine-tuning, just smaller.
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        mask = (x != 0).float().unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        hidden = self.dropout(self.relu(self.fc1(pooled)))
        output = self.sigmoid(self.fc2(hidden))
        return output.squeeze(-1)


model = SentimentClassifier(vocab_size=len(vocab)).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n  Model created!")
print(f"  Total parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024:.1f} KB")
print(f"\n  Architecture:")
print(f"    Embedding:  {len(vocab)} words x 64 dims")
print(f"    Hidden:     64 -> 32 neurons")
print(f"    Output:     32 -> 1 (positive/negative)")
print(f"\n  For comparison:")
print(f"    GPT-2:     124 MILLION parameters")
print(f"    Llama 7B:  7 BILLION parameters")
print(f"    Our model: {total_params:,} parameters (tiny but teaches the concept!)")


# ============================================================
# PART 4: Training Loop
# ============================================================
print("\n" + "=" * 60)
print("  PART 3: Training -- Watch the Model Learn!")
print("=" * 60)

X_train = torch.stack([text_to_tensor(text, vocab) for text, _ in TRAIN_DATA]).to(device)
y_train = torch.tensor([label for _, label in TRAIN_DATA], dtype=torch.float32).to(device)
X_test = torch.stack([text_to_tensor(text, vocab) for text, _ in TEST_DATA]).to(device)
y_test = torch.tensor([label for _, label in TEST_DATA], dtype=torch.float32).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print(f"\n  Training: {len(TRAIN_DATA)} examples | Testing: {len(TEST_DATA)} examples")
print(f"  Loss: Binary Cross Entropy | Optimizer: Adam (lr=0.01)")
print(f"\n  {'Epoch':<8} {'Loss':<12} {'Train Acc':<12} {'Test Acc':<12}")
print("  " + "-" * 44)

for epoch in range(1, 51):
    model.train()
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = ((model(X_train) > 0.5).float() == y_train).float().mean().item()
        test_acc = ((model(X_test) > 0.5).float() == y_test).float().mean().item()

    if epoch % 5 == 0 or epoch == 1:
        print(f"  {epoch:<8} {loss.item():<12.4f} {train_acc:<12.1%} {test_acc:<12.1%}")

print("  " + "-" * 44)
print(f"  Training complete!")


# ============================================================
# PART 5: Test the Model
# ============================================================
print("\n" + "=" * 60)
print("  PART 4: Testing -- Does It Actually Work?")
print("=" * 60)

model.eval()
test_reviews = [
    "this was an amazing and wonderful movie",
    "terrible film i hated every minute",
    "brilliant acting and a beautiful story",
    "worst movie i have ever watched so boring",
    "a fantastic experience from start to finish",
    "awful direction and dreadful performances",
    "absolutely loved it what a masterpiece",
    "complete waste of time so disappointing",
]

print(f"\n  Testing on new reviews:\n")
for review in test_reviews:
    tensor = text_to_tensor(review, vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(tensor).item()
    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
    bar = "#" * int(score * 20)
    print(f"  [{score:.2f}] {sentiment:>8} | {bar:<20} | '{review}'")


# ============================================================
# PART 6: Save the Model
# ============================================================
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_model.pt")
torch.save({
    "model_state": model.state_dict(),
    "vocab": vocab,
    "config": {"vocab_size": len(vocab), "embed_dim": 64, "hidden_dim": 32},
}, save_path)
print(f"\n  Model saved to: sentiment_model.pt ({os.path.getsize(save_path) / 1024:.1f} KB)")


# ============================================================
# PART 7: Interactive Mode
# ============================================================
print("\n" + "=" * 60)
print("  INTERACTIVE MODE -- Type a movie review!")
print("  Type 'quit' to exit")
print("=" * 60)

while True:
    try:
        review = input("\n  Your review: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not review or review.lower() in ["quit", "exit", "q"]:
        break

    tensor = text_to_tensor(review, vocab).unsqueeze(0).to(device)
    with torch.no_grad():
        score = model(tensor).item()
    sentiment = "POSITIVE" if score > 0.5 else "NEGATIVE"
    confidence = abs(score - 0.5) * 200
    print(f"  -> {sentiment} (confidence: {confidence:.0f}%, score: {score:.3f})")

print("\n" + "=" * 60)
print("  KEY TAKEAWAYS:")
print("  1. Tokenization: text -> numbers (vocabulary mapping)")
print("  2. Embedding: numbers -> dense vectors (learned during training)")
print("  3. Training loop: forward -> loss -> backward -> update weights")
print("  4. This EXACT process is used for fine-tuning GPT/BERT/Llama")
print("  5. Only difference: bigger models, more data, transformer architecture")
print("  6. Your RTX 4070 can fine-tune models up to 7B with LoRA!")
print("=" * 60)
