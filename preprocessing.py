import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

ps = PorterStemmer()


def clean_text(text):
    """
    Complete text preprocessing pipeline.
    """

    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 5. Tokenize
    words = text.split()

    # 6. Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # 7. Stemming
    words = [
        ps.stem(word)
        for word in words
    ]

    return " ".join(words)