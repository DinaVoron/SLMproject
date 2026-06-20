import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


class SmallLanguageModel:
    def __init__(
        self,
        base_model_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
        gen_params: Optional[Dict[str, Any]] = None
    ):
        """
        Инициализация модели.
        Если передан config (словарь из JSON), он переопределяет все параметры.
        """
        # Извлекаем параметры из config, если он передан
        if config:
            model_cfg = config.get("model", {})
            base_model_name = model_cfg.get("base_name", base_model_name)
            adapter_path = model_cfg.get("adapter_path", adapter_path)
            device = model_cfg.get("device", device)
            use_4bit = model_cfg.get("use_4bit", False)
            # Параметры генерации из секции "generation"
            self.gen_params = config.get("generation", {})
        else:
            use_4bit = False
            self.gen_params = gen_params or {}

        # Значения по умолчанию для параметров генерации (если не заданы)
        default_gen_params = {
            "max_new_tokens": 256,
            "temperature": 0.3,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "top_p": None,
            "top_k": None
        }
        for key, default_value in default_gen_params.items():
            if key not in self.gen_params or self.gen_params[key] is None:
                self.gen_params[key] = default_value

        self.device = device

        # Токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Конфигурация квантования (4-bit)
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        # Загрузка базовой модели
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Загрузка адаптера LoRA, если указан
        if adapter_path:
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.to(self.device)
        self.model.eval()

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096  # можно вынести в конфиг
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.gen_params.get("max_new_tokens", 256),
                temperature=self.gen_params.get("temperature", 0.3),
                do_sample=self.gen_params.get("do_sample", True),
                repetition_penalty=self.gen_params.get("repetition_penalty", 1.1),
                top_p=self.gen_params.get("top_p"),
                top_k=self.gen_params.get("top_k"),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Извлекаем только ответ ассистента (в шаблоне Vikhr)
        if "<|assistant|>" in generated:
            return generated.split("<|assistant|>")[-1].strip()
        # fallback – удаляем промпт
        return generated[len(prompt):].strip()