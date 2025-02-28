import requests
from qa_project.core.config import Config

from typing import List, Optional
from langchain.llms import BaseLLM
from langchain.schema import LLMResult, Generation

class DeepSeekClient(BaseLLM):
    """Cliente personalizado para DeepSeek"""
    
    def __init__(self):
        super().__init__()
        self._config = Config()
    
    @property
    def config(self):
        return self._config
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def _call(self, prompt: str, **kwargs) -> str:
        return self._generate([prompt], **kwargs).generations[0][0].text
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> LLMResult:
        generations = []
        
        for prompt in prompts:
            try:
                response = requests.post(
                    url="https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.config.DEEPSEEK_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": self.config.TEMPERATURE,
                        "max_tokens": self.config.MAX_TOKENS
                    }
                )
                response.raise_for_status()
                text = response.json()['choices'][0]['message']['content']
                generations.append([Generation(text=text)])
                
            except Exception as e:
                generations.append([Generation(text=f"Error: {str(e)}")])
        
        return LLMResult(generations=generations)