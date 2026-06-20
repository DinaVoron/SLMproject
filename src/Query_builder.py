from typing import List, Dict


class QueryBuilder:
    def __init__(self, max_context_tokens: int = 2000, template: str = None):
        self.max_context_tokens = max_context_tokens
        self.template = template or "<|user|>\nКонтекст: {context}\n\nВопрос: {question}\n<|end|>\n<|assistant|>\n"

    def truncate_context(self, context: str, max_tokens: int, tokenizer=None) -> str:
        """Обрезает контекст по токенам, если передан tokenizer, иначе по символам."""
        if tokenizer:
            tokens = tokenizer.encode(context)
            if len(tokens) > max_tokens:
                return context[:max_tokens - 3] + "..."
            else:
                return context

    def build(self, question: str, contexts: List[str], history: List[Dict[str, str]], tokenizer=None) -> str:
        # формируем историю диалога
        history_str = ""
        if len(history) > 0:
            history_str += history[0]
        for turn in history[-3:]:  # последние 3 раунда
            history_str += f"<|user|>{turn['user']}<|end|>\n<|assistant|>{turn['assistant']}<|end|>\n"

        combined_context = "\n\n".join(contexts)
        combined_context = self.truncate_context(combined_context, self.max_context_tokens, tokenizer)

        prompt = self.template.format(context=combined_context, question=question)
        if history_str:
            # можно добавить историю перед вопросом
            prompt = history_str + prompt
        return prompt