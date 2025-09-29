from transformers import pipeline
import os

DEFAULT_MODEL = 'google/flan-t5-small'

class LLMWrapper:
    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        # Use text2text-generation pipeline for flan-style models
        self.generator = pipeline('text2text-generation', model=self.model_name, max_length=self.max_length)

    def answer(self, prompt: str) -> str:
        # Keep prompt length reasonable for small models
        out = self.generator(prompt, max_length=self.max_length, do_sample=False)
        return out[0]['generated_text']

if __name__ == '__main__':
    wrapper = LLMWrapper()
    print(wrapper.answer('Summarize: The electrification of rural areas faces technical and economic barriers.'))
