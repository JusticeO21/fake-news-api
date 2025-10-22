import re
import spacy

nlp = spacy.load("en_core_web_sm")

def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()

def remove_punctuation(text: str) -> str:
    """Remove punctuation and special characters."""
    return re.sub(r"[^\w\s]", "", text)

def lemmatize_and_remove_stopwords(text: str) -> str:
    """Tokenize, lemmatize, and remove stopwords using spaCy."""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return " ".join(tokens)

def clean_text(text: str) -> str:
    """
    Complete cleaning pipeline:
    1. Lowercase
    2. Remove punctuation
    3. Lemmatize + remove stopwords
    """
    text = lowercase_text(text)
    text = remove_punctuation(text)
    text = lemmatize_and_remove_stopwords(text)
    return text
