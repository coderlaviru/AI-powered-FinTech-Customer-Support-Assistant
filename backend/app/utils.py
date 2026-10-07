#backend/app/utils.py
import re


def clean_text(text: str) -> str:
    """Normalize extracted document text without changing its meaning."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()
