from collections import Counter, defaultdict

# ---------------------------------------------------------
# Exercise 1: Generate Bigrams and Trigrams
# ---------------------------------------------------------
print("--- Exercise 1 ---")
sentence = "I love natural language processing"
tokens = sentence.lower().split()

def get_ngrams(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

bigrams = get_ngrams(tokens, 2)
trigrams = get_ngrams(tokens, 3)

print(f"Sentence: {sentence}")
print(f"Bigrams: {bigrams}")
print(f"Trigrams: {trigrams}")
print()

# ---------------------------------------------------------
# Exercise 2: MLE Probability
# ---------------------------------------------------------
print("--- Exercise 2 ---")
corpus = ["i love nlp", "i love python", "i hate bugs", "nlp is great"]

# Standardize and tokenize the corpus
flat_tokens = []
all_bigrams = []

for s in corpus:
    s_tokens = s.lower().split()
    flat_tokens.extend(s_tokens)
    for i in range(len(s_tokens) - 1):
        all_bigrams.append((s_tokens[i], s_tokens[i+1]))

# Count occurrences
unigram_counts = Counter(flat_tokens)
bigram_counts = Counter(all_bigrams)

# Calculate P(love | i) = count(i, love) / count(i)
count_i = unigram_counts['i']
count_i_love = bigram_counts[('i', 'love')]
prob_love_given_i = count_i_love / count_i if count_i > 0 else 0

print(f"Count('i'): {count_i}")
print(f"Count('i', 'love'): {count_i_love}")
print(f"P(love | i) = {prob_love_given_i}")
print()

# ---------------------------------------------------------
# Exercise 3: Bigram Predicter
# ---------------------------------------------------------
print("--- Exercise 3 ---")
text_corpus = "the cat sat on the mat . the dog sat on the rug . the cat is on the mat ."
tokens_ex3 = text_corpus.lower().split()

def build_bigram_model(tokens):
    model = defaultdict(lambda: Counter())
    for i in range(len(tokens) - 1):
        model[tokens[i]][tokens[i+1]] += 1
    
    # Convert counts to probabilities
    prob_model = {}
    for word, neighbors in model.items():
        total_count = sum(neighbors.values())
        prob_model[word] = {next_w: count / total_count for next_w, count in neighbors.items()}
    return prob_model

def predict_next(model, word):
    word = word.lower()
    if word in model:
        # Get the word with the highest probability
        return max(model[word], key=model[word].get)
    return "None (Word not in model)"

model_ex3 = build_bigram_model(tokens_ex3)
test_word = "the"
prediction = predict_next(model_ex3, test_word)

print(f"Corpus: {text_corpus}")
print(f"Prediction for '{test_word}': {prediction}")
print()

# ---------------------------------------------------------
# Exercise 4: Add-1 (Laplace) Smoothing
# ---------------------------------------------------------
print("--- Exercise 4 ---")

def build_bigram_model_smoothed(tokens):
    vocab = set(tokens)
    V = len(vocab)
    
    # Count bigrams and unigrams
    unigram_counts = Counter(tokens)
    bigram_counts = Counter()
    for i in range(len(tokens) - 1):
        bigram_counts[(tokens[i], tokens[i+1])] += 1
    
    # Function to get smoothed probability
    def get_prob(w1, w2):
        # P(w2|w1) = (count(w1,w2) + 1) / (count(w1) + V)
        c_w1_w2 = bigram_counts[(w1, w2)]
        c_w1 = unigram_counts[w1]
        return (c_w1_w2 + 1) / (c_w1 + V)
    
    return get_prob, vocab

get_prob_smooth, vocab_ex4 = build_bigram_model_smoothed(tokens_ex3)
w1, w2 = "the", "cat"
smooth_prob = get_prob_smooth(w1, w2)

print(f"Smoothed P({w2} | {w1}) = {smooth_prob}")
