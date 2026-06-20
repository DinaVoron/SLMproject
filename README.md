Данный модуль разработан для использования совместно с модулем Диалога и модулем Индексации документов.
### Инструкцию по развёртыванию
1. Поместите папку с другими модулями в корень проекта.
2. Отредактируйте config.json.
- device: "cuda" (если есть NVIDIA GPU) или "cpu"
- use_4bit: true (только для GPU) / false (для CPU)
- use_lemmatizer: / false
### Структура проекта
```├── config.json          
├── qa_adapter/          
└── src/
    ├── main.py
    ├── SLM.py          
    ├── Dialogue_manager.py
    ├── Query_builder.py
    ├── Preprocessor.py
    ├── Postprocessor.py
    ├── Retriever.py
    └── ConfigLoader.py```
