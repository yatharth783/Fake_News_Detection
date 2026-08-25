import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =========================================================
# NLTK STOPWORDS
# =========================================================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))


# =========================================================
# STEMMER
# =========================================================

ps = PorterStemmer()


# =========================================================
# TEXT PREPROCESSING FUNCTION
# =========================================================

def clean_text(text):
    """
    Cleans and preprocesses news text before
    sending it to the machine learning model.

    Steps:
    1. Validate input
    2. Convert to lowercase
    3. Remove URLs
    4. Remove punctuation
    5. Remove numbers
    6. Remove stopwords
    7. Apply stemming
    8. Remove extra spaces
    """

    # -----------------------------------------------------
    # 1. Validate input
    # -----------------------------------------------------

    if not isinstance(text, str):
        return ""

    if not text.strip():
        return ""


    # -----------------------------------------------------
    # 2. Convert to lowercase
    # -----------------------------------------------------

    text = text.lower()


    # -----------------------------------------------------
    # 3. Remove URLs
    # -----------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        "",
        text
    )


    # -----------------------------------------------------
    # 4. Remove punctuation
    # -----------------------------------------------------

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )


    # -----------------------------------------------------
    # 5. Remove numbers
    # -----------------------------------------------------

    text = re.sub(
        r"\d+",
        "",
        text
    )


    # -----------------------------------------------------
    # 6. Split into words
    # -----------------------------------------------------

    words = text.split()


    # -----------------------------------------------------
    # 7. Remove stopwords
    # -----------------------------------------------------

    words = [
        word
        for word in words
        if word not in stop_words
    ]


    # -----------------------------------------------------
    # 8. Apply stemming
    # -----------------------------------------------------

    words = [
        ps.stem(word)
        for word in words
    ]


    # -----------------------------------------------------
    # 9. Remove empty values
    # -----------------------------------------------------

    words = [
        word
        for word in words
        if word.strip()
    ]


    # -----------------------------------------------------
    # 10. Return cleaned text
    # -----------------------------------------------------

    return " ".join(words).strip()