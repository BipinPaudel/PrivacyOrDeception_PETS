from src.utils.initialization import read_config_from_yaml
from data.reddit import load_data, create_prompts
import argparse
from src.synthetic.synthetic_final import run_synthetic_final
from src.configs.config import Task,Experiment

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process inputs") #  EXPERIMENT_EXPLICIT_SANITIZATION EXPERIMENT_POST_INFERENCE, EXPERIMENT_LLM_UTILITY
    parser.add_argument('--experiment', type=str, default=[Experiment.EXPERIMENT_SANITIZATION.value], help='The type of experiment')
    # features = income, age, gender, married,
    # features = age, sex, relationship_status, income_level
    # ['sex', 'relationship_status','income_level', 'age']
    parser.add_argument('--feature', type=str, default=['income_level','sex','age','relationship_status',], help='Private feature of person')
    parser.add_argument('--hardness', type=list, default=[1,2,3,4,5], help='how hard it was')
    args = parser.parse_args()
    

    # env = "configs/reddit_llama3.1_70b.yaml"
    # env = "configs/reddit_llama3.1_8b.yaml"
    # env = "configs/nautilus_deepseek32b.yaml"
    # env = "configs/nautilus_llama3.yaml"
    env = "configs/priv_gpt3.5.yaml"
    
    cfg = read_config_from_yaml(env)
    
    print('Configuration setup done')
    
    if cfg.task == Task.SYNTHETIC:
        # run_synthetic(cfg, Experiment.TOPIC_PRIOR.value, 'sex', args.hardness)
        for exp in args.experiment:
            run_synthetic_final(cfg, exp, '', args.hardness)
        