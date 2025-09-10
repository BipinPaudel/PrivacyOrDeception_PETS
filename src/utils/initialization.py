from src.configs import Config
from pydantic import ValidationError
import yaml

def read_config_from_yaml(path) -> Config:
    with open(path, 'r') as stream:
        try:
            yaml_obj = yaml.safe_load(stream)
            cfg = Config(**yaml_obj)
            return cfg
        except (yaml.YAMLError, ValidationError) as exception:
            print(exception)
            raise exception
        