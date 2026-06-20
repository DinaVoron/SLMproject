from typing import List, Dict
from Preprocessor import QuestionPreprocessor
from Retriever import RAGRetriever
from Query_builder import QueryBuilder
from SLM import SmallLanguageModel
from Postprocessor import ResponsePostprocessor


class DialogueManager:
    def __init__(self, preprocessor: QuestionPreprocessor, retriever: RAGRetriever,
                 query_builder: QueryBuilder, slm: SmallLanguageModel,
                 postprocessor: ResponsePostprocessor, max_history_turns: int = 10):
        self.preprocessor = preprocessor
        self.retriever = retriever
        self.query_builder = query_builder
        self.slm = slm
        self.postprocessor = postprocessor
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_to_history(self, question: str, answer: str) -> None:
        self.history.append({"user": question, "assistant": answer})
        if len(self.history) > self.max_history_turns:
            self.history.pop(0)

    def get_history(self, max_tokens: int = 2000) -> List[Dict[str, str]]:
        return self.history

    def clear_history(self) -> None:
        self.history.clear()

    def answer_question(self, question: str) -> str:
        cleaned = self.preprocessor.clean_text(question)
        chunks = self.retriever.retrieve(cleaned)
        history = self.get_history()
        prompt = self.query_builder.build(cleaned, chunks, history, tokenizer=self.slm.tokenizer)
        raw = self.slm.generate(prompt)
        answer = self.postprocessor.process(raw)
        self.add_to_history(question, answer)
        return answer