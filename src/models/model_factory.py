from src.configs import ModelConfig
from .model import BaseModel
from .hf_model import HFModel
from .open_ai import OpenAIGPT
from .nautilus_model import LocalLLM

def get_model(config: ModelConfig) -> BaseModel:
    print(f'model factory: {config.provider}')
    if config.provider == "hf":
        return HFModel(config)
    elif config.provider == "openai":
        return OpenAIGPT(config)
    elif config.provider == 'nautilus':
        return LocalLLM(config)
    else:
        raise NotImplementedError
    