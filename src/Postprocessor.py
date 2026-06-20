class ResponsePostprocessor:
    def __init__(self, remove_special_tokens: bool = True):
        self.remove_special_tokens = remove_special_tokens

    def process(self, raw_response: str) -> str:
        text = raw_response
        if self.remove_special_tokens:
            text = text.replace("<|end|>", "").replace("</s>", "")
            if text.startswith("<|assistant|>"):
                text = text[len("<|assistant|>"):]
        return self.format_response(text)

    def format_response(self, response: str) -> str:
        if not response:
            return ""
        response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()
        if not response.endswith(('.', '!', '?')):
            response += '.'
        return response