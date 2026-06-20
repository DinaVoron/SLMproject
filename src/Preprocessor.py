import re
from typing import List
from pymorphy2 import MorphAnalyzer


class QuestionPreprocessor:
    def __init__(self, stop_words: List[str] = None, use_lemmatizer: bool = True):
        self.stop_words = set(stop_words or [])
        self.use_lemmatizer = use_lemmatizer
        if self.use_lemmatizer:
            self.lemmatizer = MorphAnalyzer()
        else:
            self.lemmatizer = None

    def clean_text(self, question: str) -> str:
        """Приведение к нижнему регистру, удаление лишних символов."""
        text = question.lower().strip()
        text = re.sub(r'[^\w\s?]', '', text)
        return text

    def lemmatize(self, question: str) -> str:
        if not self.lemmatizer:
            return question
        words = question.split()
        lemmatized = [self.lemmatizer.parse(word)[0].normal_form for word in words]
        return ' '.join(lemmatized)

    def extract_keywords(self, question: str) -> List[str]:
        cleaned = self.clean_text(question)
        if self.use_lemmatizer:
            cleaned = self.lemmatize(cleaned)
        tokens = cleaned.split()
        keywords = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        return keywords