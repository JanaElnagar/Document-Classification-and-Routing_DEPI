"""Text cleaning shared by training and inference (FR8).

Keep this module the single place where text is cleaned, so the model
sees the same input at training time and in the API.
"""
import re

MAX_CHARS = 3000  # long documents are cut; the start carries most of the signal

# Markers that reveal which public dataset a document came from. If we keep them,
# a model can learn "which source" instead of "which category" (risk R5).
_CFPB_MASK = re.compile(r"\b(?:XX+/?)+(?:\d{2,4})?\b")      # XXXX, XX/XX/2019
_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")                 # Bitext {{Order Number}}
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_REPEATED_PUNCT = re.compile(r"([=*_\-~#.])\1{3,}")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text) -> str:
    """Normalize one document's text. Case is kept; TF-IDF lowercases on its own."""
    if text is None:
        return ""
    text = str(text)
    text = _CFPB_MASK.sub(" ", text)
    text = _PLACEHOLDER.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _REPEATED_PUNCT.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text[:MAX_CHARS]
