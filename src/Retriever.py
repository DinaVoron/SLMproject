from typing import List


class RAGRetriever:
    def __init__(self, top_k: int = 5, embedding_model=None, vector_store=None):
        self.top_k = top_k
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(self, question: str) -> List[str]:
        """Возвращает список текстовых фрагментов, релевантных вопросу."""
        # Заглушка – в реальности вызывает эмбеддинги и поиск
        return ["Документ 1. Раздел 1. Котики очень милые", "Документ 2. Раздел 3. Собачки, в отличие от котиков, веселые."]
