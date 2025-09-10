from langchain.llms.base import LLM
from openai import OpenAI
from pydantic import Field
from typing import Optional, List
from langchain.embeddings.base import Embeddings
import asyncio
from src.configs import ModelConfig
class LocalLLM(LLM):
    """
    A LangChain-compatible wrapper for the OpenAI client (local LM Studio instance).
    """

    # base_url: str = Field(..., description="Base URL for the LM Studio server")
    # api_key: str = Field(..., description="API key for LM Studio authentication")
    # model_name: str = Field(..., description="Name of the model to use")
    # temperature: float = Field(0.7, description="Sampling temperature")
    # max_tokens: int = Field(256, description="Maximum number of tokens to generate")
    config: ModelConfig = Field(..., description="Model configuration")
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, config: ModelConfig, **kwargs):
        kwargs['config'] = config
        super().__init__(**kwargs)
        # Initialize the OpenAI client (store it separately)
        # self.config = config
        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 2000
        
        self._client = OpenAI(
            base_url=self.config.args['base_url'],
            api_key=self.config.args['api_key'],
            )
        

    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Call the LM Studio API to generate a response for the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            stop (Optional[List[str]]): Optional stop tokens for the model.

        Returns:
            str: The generated response.
        """
        # system_prompt = "You are an assistant that delivers precise and formal responses to questions, utilizing the provided context."
        # system_prompt = kwargs.get('system_prompt', system_prompt)
            
        response = self._client.chat.completions.create(
            model=self.config.args['model_name'],
            messages=[
                # {"role": "system",  "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.args['temperature'],
            max_tokens=self.config.args['max_tokens'],
            stop=stop,
        )
        return response.choices[0].message.content
        # return response
        
    def call_with_metadata(self, prompt: str, stop: Optional[List[str]] = None):
        response = self._client.chat.completions.create(
            model=self.config.args['model_name'],
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.args['temperature'],
            max_tokens=self.config.args['max_tokens'],
            stop=stop,
        )
        return response.choices[0].message.content, response.usage
    
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """
        Asynchronous call to the LM Studio API to generate a response for the given prompt.

        Args:
            prompt (str): The input prompt for the model.
            stop (Optional[List[str]]): Optional stop tokens for the model.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The generated response.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, self._call, prompt, stop, **kwargs
        )