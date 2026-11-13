
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Input sentence
sentence = "NLP techniques are used in virtual assistants like Alexa and Siri."

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Step 1: Tokenization
tokens = word_tokenize(sentence)

# Step 2: Remove stopwords
filtered_tokens = [
    word for word in tokens
    if word.lower() not in stop_words
]

# Step 3: Apply stemming
stemmed_words = [
    stemmer.stem(word)
    for word in filtered_tokens
    if word.isalpha()
]

# Print results
print("Original Tokens:", tokens)
print("Tokens Without Stopwords:", filtered_tokens)
print("Stemmed Words:", stemmed_words)
