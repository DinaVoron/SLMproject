from src.Preprocessor import QuestionPreprocessor
from src.Retriever import RAGRetriever
from src.Query_builder import QueryBuilder
from src.SLM import SmallLanguageModel
from src.Postprocessor import ResponsePostprocessor
from src.Dialogue_manager import DialogueManager
from src.ConfigLoader import load_config


def main():
    print("Загружаем параметры генерации...")
    cfg = load_config()

    print("Инициализируем предобработчик вопросов...")
    preproc = QuestionPreprocessor(
        stop_words=cfg["preprocessor"].get("stop_words", []),
        use_lemmatizer=cfg["preprocessor"].get("use_lemmatizer", True)
    )

    print("Инициализируем retriever...")
    retriever = RAGRetriever(
        top_k=cfg["retriever"].get("top_k", 5),
        embedding_model=cfg["retriever"].get("embedding_model"),
        vector_store=cfg["retriever"].get("index_path")
    )

    print("Инициализируем построитель запросов...")
    qbuilder = QueryBuilder(
        max_context_tokens=cfg["query_builder"].get("max_context_tokens",
                                                    2000),
        template=cfg["query_builder"].get("template")
    )

    print("Инициализируем малую языковую модель...")
    slm = SmallLanguageModel(
        base_model_name=cfg["model"]["base_name"],
        adapter_path=cfg["model"].get("adapter_path"),
        device=cfg["model"].get("device", "cpu"),
        config=cfg
    )

    print("Инициализируем постобработчик ответов...")
    postproc = ResponsePostprocessor(
        remove_special_tokens=cfg["postprocessor"].get("remove_special_tokens", True)
    )
    print("Инициализируем диалоговый менеджер...")
    dm = DialogueManager(
        preprocessor=preproc,
        retriever=retriever,
        query_builder=qbuilder,
        slm=slm,
        postprocessor=postproc,
        max_history_turns=cfg["dialogue_manager"].get("max_history_turns", 10)
    )

    print("RAG-ассистент готов. Введите 'exit' для выхода.")
    while True:
        q = input("\nВаш вопрос: ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue
        ans = dm.answer_question(q)
        print(f"Ответ: {ans}")


if __name__ == "__main__":
    main()