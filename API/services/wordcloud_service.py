
from collections import Counter



def generate_wordcloud(comments):
    custom_stopwords = {
        "this", "that", "with", "from", "video",
        "have", "they", "your", "just", "about"
    }

    all_text = " ".join(comments)
    words = all_text.lower().split()

    words = [
        word
        for word in words
        if len(word) > 3 and word not in custom_stopwords
    ]

    word_counts = Counter(words)

    return word_counts.most_common(50)