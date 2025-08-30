# backend/llm_selector.py
from typing import Any, Optional
import os
from dotenv import load_dotenv

load_dotenv()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "hf").lower()
LOCAL_LLM_PATH = os.getenv("LOCAL_LLM_PATH", "./models/my_local_llm")

def get_llm():
    """
    Returns an object compatible with langchain LLM interface.
    Supports:
      - local HF sequence-to-sequence via pipeline (transformers)
      - (optionally) OpenAI chat if LLM_PROVIDER == "openai"
    """
    if LLM_PROVIDER == "openai":
        # If you ever re-enable OpenAI:
        from langchain_community.chat_models import ChatOpenAI
        return ChatOpenAI(temperature=0.0)
    else:
        # Use a local HF model via the HF pipeline
        # This assumes you have a local model saved at LOCAL_LLM_PATH or a model id
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
        model_path = LOCAL_LLM_PATH
        if not os.path.exists(model_path):
            # if model path doesn't exist, try to use a hub id from env (fallback)
            hub_id = os.getenv("HF_LLM_ID")
            if hub_id:
                model_path = hub_id
            else:
                raise RuntimeError(
                    f"No local LLM found at {LOCAL_LLM_PATH} and HF_LLM_ID not set."
                )

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=0 if os.getenv("USE_GPU") else -1, max_length=512)
        
        # Wrap into a tiny object with .generate or .__call__ compatible with langchain
        class HFLLMWrapper:
            def __init__(self, pipe):
                self.pipe = pipe
            def __call__(self, prompt: str, **kwargs):
                out = self.pipe(prompt, **kwargs)
                # pipeline returns list of dicts
                return out[0]["generated_text"]
            # For compatibility with some langchain LLMChain usage:
            def generate(self, prompts, **kwargs):
                texts = [p for p in prompts]
                results = [self.pipe(t, **kwargs)[0]["generated_text"] for t in texts]
                # naive structure:
                return {"generations": [[{"text": r}] for r in results]}

        return HFLLMWrapper(pipe)
