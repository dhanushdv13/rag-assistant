import os
import torch
import requests
from typing import Optional, List, Any
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain_huggingface import HuggingFacePipeline

from .config import settings

def get_device():
    """Get the appropriate device for model inference"""
    return "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# Enhanced Prompt Templates
# ============================================================================

PHI3_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""<|system|>
You are a helpful AI assistant. Use the provided context to answer the question accurately and concisely. If the context doesn't contain enough information, say so clearly.
<|end|>
<|user|>
Context: {context}

Question: {question}

Please provide a clear, accurate answer based on the context above.
<|end|>
<|assistant|>"""
)

T5_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Based on the following information, provide a clear and concise answer to the question.

Information: {context}

Question: {question}

Answer:"""
)

# ============================================================================
# Enhanced Ollama Phi-3 Integration with Intelligent Fallback
# ============================================================================

class OllamaPhi3(LLM):
    """Enhanced Ollama Phi-3 with intelligent timeout handling and T5 fallback"""
    
    model_name: str = "phi3:mini"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 150

    @property
    def _llm_type(self) -> str:
        return "ollama-phi3"

    @property
    def _identifying_params(self):
        """Get the identifying parameters - FIXED with @property decorator"""
        return {"model": self.model_name, "temperature": self.temperature}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> str:
        """Enhanced call with intelligent timeout handling and T5 fallback"""
        try:
            # Try Ollama with optimized timeout
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model_name,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens,
                        'top_p': 0.9,
                        'repeat_penalty': 1.3
                    }
                },
                timeout=60  # Optimized timeout
            )
            
            if response.status_code == 200:
                text = response.json().get('response', '')
                if text and len(text.strip()) > 10:  # Valid response
                    return self._clean_response(text)
                    
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print("⚠️ Ollama timeout, using T5 fallback for this question...")
            return self._t5_fallback_generation(prompt)
        except Exception as e:
            print(f"⚠️ Ollama error: {e}, using T5 fallback...")
            return self._t5_fallback_generation(prompt)
        
        # If we get here, Ollama failed - use T5 fallback
        return self._t5_fallback_generation(prompt)

    def _t5_fallback_generation(self, prompt: str) -> str:
        """Generate response using T5 when Ollama fails"""
        try:
            # Extract question and context from prompt
            if "Question:" in prompt and "Context:" in prompt:
                parts = prompt.split("Question:")
                if len(parts) > 1:
                    question_part = parts[1].strip()
                    # Get just the question (before any assistant tags)
                    question = question_part.split("<|end|>")[0].strip()
                    
                    # Extract context
                    context_parts = prompt.split("Context:")
                    if len(context_parts) > 1:
                        context = context_parts[1].split("Question:")[0].strip()
                        
                        # Create optimized prompt for T5
                        t5_prompt = f"Based on this information, answer the question: {question}\n\nInformation: {context[:1000]}"  # Limit context length
                        
                        # Use T5 pipeline with optimized parameters
                        pipe = pipeline(
                            "text2text-generation",
                            model="google/flan-t5-base",
                            max_new_tokens=100,
                            do_sample=True,
                            temperature=0.7,
                            repetition_penalty=1.5
                        )
                        
                        result = pipe(t5_prompt)[0]['generated_text']
                        return f"Based on the provided information: {result}"
                        
            return "I found relevant information in your document, but had technical difficulties generating a complete response. Please try asking a more specific question."
            
        except Exception as e:
            return "I found relevant information but encountered technical difficulties. The document contains relevant details about your question."

    def _clean_response(self, text: str) -> str:
        """Clean and validate the response from Ollama"""
        if not text or not text.strip():
            return "I couldn't generate a response based on the provided context."
        
        # Remove common artifacts
        clean_text = text.strip()
        artifacts = ["<|assistant|>", "<|user|>", "<|system|>", "<|end|>"]
        for artifact in artifacts:
            clean_text = clean_text.replace(artifact, "").strip()
        
        # Remove leading punctuation
        clean_text = clean_text.lstrip(':').lstrip('-').strip()
        
        # Limit length and ensure quality
        if len(clean_text) > 500:
            sentences = clean_text.split('. ')
            if len(sentences) > 1:
                clean_text = '. '.join(sentences[:3]) + '.'
            else:
                clean_text = clean_text[:500] + '...'
        
        return clean_text

# ============================================================================
# Model Loading Functions
# ============================================================================

def try_ollama_phi3():
    """Try to connect to Ollama Phi-3 with comprehensive testing"""
    try:
        print("🦙 Connecting to Ollama Phi-3...")
        
        # Test 1: Basic connection
        health_response = requests.get("http://localhost:11434", timeout=10)
        if health_response.status_code != 200:
            print(f"❌ Ollama not responding: {health_response.status_code}")
            return None
        
        # Test 2: Check available models
        model_response = requests.get("http://localhost:11434/api/tags", timeout=10)
        models = model_response.json().get('models', [])
        model_names = [m.get('name', '') for m in models]
        
        if not any('phi3' in name.lower() for name in model_names):
            print(f"❌ Phi-3 not found. Available models: {model_names}")
            return None
        
        # Test 3: Generation test
        test_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3:mini",
                "prompt": "Hello",
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=30
        )
        
        if test_response.status_code == 200:
            test_result = test_response.json().get('response', '')
            print(f"✅ Ollama Phi-3 test successful: {test_result[:50]}...")
            return OllamaPhi3()
        else:
            print(f"❌ Phi-3 generation test failed: {test_response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Failed to connect to Ollama Phi-3: {e}")
        return None

def get_t5_fallback():
    """Enhanced T5 fallback with better generation parameters"""
    try:
        print("🤖 Loading enhanced T5 fallback...")
        
        # Use FLAN-T5 which is better for Q&A
        model_name = "google/flan-t5-base"
        
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        print("✅ Enhanced T5 fallback loaded successfully!")
        return HuggingFacePipeline(pipeline=pipe)
        
    except Exception as e:
        print(f"❌ Enhanced T5 failed, using T5-small: {e}")
        
        # Ultra-minimal fallback
        pipe = pipeline(
            "text2text-generation",
            model="t5-small",
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.5
        )
        
        print("✅ T5-small fallback loaded!")
        return HuggingFacePipeline(pipeline=pipe)

def get_llm():
    """Load LLM with Phi-3 priority and intelligent fallbacks"""
    
    # Try Ollama Phi-3 first
    phi3_model = try_ollama_phi3()
    if phi3_model:
        return phi3_model, PHI3_PROMPT
    
    # Fallback to enhanced T5
    print("🔄 Using T5 fallback...")
    t5_model = get_t5_fallback()
    return t5_model, T5_PROMPT

# ============================================================================
# Response Quality Control
# ============================================================================

def clean_generated_answer(text: str) -> str:
    """Clean and validate generated responses"""
    if not text or not text.strip():
        return "I couldn't generate a useful answer based on the available context."
    
    clean_text = text.strip()
    
    # Remove common assistant/user tokens
    tokens_to_remove = ["<|assistant|>", "<|user|>", "<|system|>", "<|end|>", "Assistant:", "User:"]
    for token in tokens_to_remove:
        if clean_text.startswith(token):
            clean_text = clean_text[len(token):].strip()
    
    # Check for hallucination indicators
    hallucination_patterns = [
        "truestrength", "false unable", "sequel", "tasten", "tode",
        "denial", "###", "---", "...", "null", "undefined"
    ]
    
    if any(pattern.lower() in clean_text.lower() for pattern in hallucination_patterns):
        return "I couldn't find specific information to answer your question. Please try rephrasing or check if the relevant information is available in your documents."
    
    # Check for excessive repetition
    words = clean_text.split()
    if len(words) > 5:
        unique_words = len(set(words))
        if unique_words / len(words) < 0.3:  # Too much repetition
            return "I'm having difficulty generating a clear response. Please try asking a more specific question."
    
    # Limit length to reasonable size
    if len(clean_text) > 2000:
        clean_text = clean_text[:2000] + "..."
    
    return clean_text

# ============================================================================
# QA Chain Builder
# ============================================================================

def build_qa_chain(vectorstore):
    """Build QA chain with enhanced model selection"""
    if vectorstore is None:
        raise ValueError("Vectorstore is required")
    
    # Create retriever with optimal settings
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )
    
    # Get the best available LLM and prompt
    llm, prompt_template = get_llm()
    
    # Create the QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": prompt_template,
            "document_separator": "\n\n---\n\n"
        },
        return_source_documents=True,
        verbose=False
    )
    
    return chain

# ============================================================================
# Utility and Testing Functions
# ============================================================================

def test_model_generation():
    """Test the current model setup"""
    try:
        llm, prompt = get_llm()
        
        if isinstance(llm, OllamaPhi3):
            test_prompt = "What is machine learning?"
            response = llm._call(test_prompt)
            print(f"✅ Model test successful: {response[:100]}...")
            return True
        else:
            # Test HuggingFace pipeline
            test_input = ["What is machine learning?"]
            result = llm.generate(test_input)
            response = result.generations[0][0].text
            cleaned = clean_generated_answer(response)
            print(f"✅ Model test successful: {cleaned[:100]}...")
            return True
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def get_model_info():
    """Get information about the currently loaded model"""
    return {
        "device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "ollama_status": "checking...",
        "model_type": "phi3 with t5 fallback"
    }

def validate_response_quality(text: str, question: str) -> str:
    """Enhanced response validation"""
    if not text or len(text.strip()) < 10:
        return "I need more context to provide a complete answer."
    
    text = text.strip()
    
    # Check for timeout errors
    if "timeout" in text.lower() or "error:" in text.lower():
        return "I found relevant information in your document but had technical difficulties. Please try rephrasing your question."
    
    # Check for common hallucination patterns
    hallucination_indicators = [
        "truestrength", "false unable", "sequel", "tasten", "tode",
        "denial", "###", "---", "null", "undefined"
    ]
    
    if any(indicator.lower() in text.lower() for indicator in hallucination_indicators):
        return f"I found relevant information about '{question}' but couldn't generate a clear response. Please try asking a more specific question."
    
    return text

# ============================================================================
# Memory Optimization
# ============================================================================

def optimize_memory():
    """Clean up memory usage"""
    try:
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("🧹 Memory optimization completed")
        
    except Exception as e:
        print(f"⚠️ Memory optimization warning: {e}")

# ============================================================================
# Initialize
# ============================================================================

# Initialize memory optimization
optimize_memory()
