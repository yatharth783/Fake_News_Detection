import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords
#nltk.download('stopwords')
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
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # 4. Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 5. Remove unnecessary whitespace and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 6. Tokenization, Stopword removal, and Stemming
    words = text.split()
    words = [ps.stem(w) for w in words if w not in stop_words]
    
    return " ".join(words)

# Example for demonstration (as requested in requirements)
if __name__ == "__main__":
    sample = "Check out our website https://fake-news.com! <b>Breaking News:</b> The government is hiding aliens!!"
    print(f"BEFORE: {sample}")
    print(f"AFTER:  {clean_text(sample)}")