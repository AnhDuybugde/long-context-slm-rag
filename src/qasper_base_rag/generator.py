from __future__ import annotations

from .chunking import Chunk


class SmallSeq2SeqGenerator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def answer(
        self,
        question: str,
        contexts: list[Chunk],
        *,
        max_input_tokens: int = 1024,
        max_new_tokens: int = 96,
    ) -> str:
        context_text = "\n\n".join(
            f"[{index + 1}] Title: {chunk.title}\nSection: {chunk.section}\n{chunk.text}"
            for index, chunk in enumerate(contexts)
        )
        prompt = (
            "Answer the question using only the provided context. "
            "If the answer is not in the context, answer Unanswerable.\n\n"
            f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

