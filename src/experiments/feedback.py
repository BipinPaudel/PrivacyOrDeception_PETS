import tqdm
from data.reddit import (create_baseline_prompt,create_sanitization_prompt,
                         load_data, load_json_obj_from_file,write_json_lists_to_file,
                         create_feedback_inference_prompt,create_feedback_anonymizer_prompt, create_sentence_similarity_prompt)
from data.filter import get_unique_private_attribute
from src.reddit.reddit_utils import extract_inference_from_response, extract_anonymized_txt_from_response, extract_last_anonymized_text

from src.models.model_factory import get_model
    
def infer_model(cfg, feature, hardness, iteration):
    
    filename =  f'{feature}_comments.jsonl'
    comments = load_json_obj_from_file(cfg.task_config.path+filename)
    if iteration  > 1:
        new_filename = f"results/{iteration-1}/sanitization_anonymizer_{iteration-1}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
        new_comments = load_json_obj_from_file(new_filename)
        
    
    private_values = get_unique_private_attribute(comments, feature)
    prompts = []
    prompts_to_write = []
    for i, comment in enumerate(comments):
        if iteration > 1:
            comment['text'] = extract_anonymized_txt_from_response(new_comments[i]['anonymize_result'], new_comments[i]['id'])
        prompt = create_feedback_inference_prompt(comment, (feature, private_values))
        prompts.append(prompt[0])
        prompts_to_write.append({ 'id': comment['id'], 'prompt': str(prompt[0])})
    
    write_json_lists_to_file(f'results/{iteration}/prompts_infer_model_{iteration}_{feature}.jsonl', prompts_to_write)
    print('prompt ok')
    model = get_model(cfg.gen_model)
    
    
    results = model.predict_multi(prompts, max_workers=4)
    
    
    results_temp = []
    for res in results:
        print(feature, )
        print(res[1])
        print('**************'*3)
        results_temp.append(res[1])
        
    for i,comment in enumerate(comments): # CHANGE HERE
        comment["inference_result"] = results_temp[i]
        
        
    filename = f"results/{iteration}/sanitization_inference_{iteration}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    write_json_lists_to_file(filename, comments)
    
    
def anonymize_text(cfg, feature, hardness, iteration):
    filename = f"results/{iteration}/sanitization_inference_{iteration}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    comments = load_json_obj_from_file(filename)
    
    
    prompts = []
    prompts_to_write = []
    for comment in comments:
        # print()
        inference, prediction = extract_inference_from_response(comment["inference_result"], comment['id'])
        prompt = create_feedback_anonymizer_prompt(comment, 
                                                   feature=feature, 
                                                   inference=inference,
                                                   prediction=prediction)
        prompts.append(prompt[0])
        prompts_to_write.append({ 'id': comment['id'], 'prompt': str(prompt[0])})
    write_json_lists_to_file(f'results/{iteration}/prompts_anonymize_model_{iteration}_{feature}.jsonl', prompts_to_write)
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=4)
    
    results_temp = []
    for res in results:
        results_temp.append(res[1])
        
    for i,comment in enumerate(comments): # CHANGE HERE
        comment["anonymize_result"] = results_temp[i]
        
        
    filename = f"results/{iteration}/sanitization_anonymizer_{iteration}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    write_json_lists_to_file(filename, comments)
    
def run_feedback_utility(cfg, feature, hardness, iteration, anonymized_filename):
    
    filename =  f'{feature}_comments.jsonl'
    original_comments = load_json_obj_from_file(cfg.task_config.path+filename)
    
    # anonymized_filename = f"results/{iteration}/sanitization_anonymizer_{iteration}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    anonymized_comments = load_json_obj_from_file(anonymized_filename)
    
    assert (len(original_comments) == len(anonymized_comments))
    prompts = []
    
    # original_comments = original_comments[:10]
    
    prompts_to_write = []
    for original_comment, anonymized_comment in zip(original_comments, anonymized_comments):
        assert original_comment['id'] == anonymized_comment['id']
        orgi = original_comment['text']
        # para = extract_anonymized_txt_from_response(anonymized_comment['anonymize_result'], anonymized_comment['id'])        
        para = extract_last_anonymized_text(anonymized_comment['sanitized_response'])#extract_anonymized_txt_from_response(anonymized_comment['sanitized_response'], anonymized_comment['id'])        
        prompt = create_sentence_similarity_prompt(orgi, para)
        prompts_to_write.append({ 'id': ['id'], 'prompt': str(prompt)})
        prompts.append(prompt)
        
    # print(f"Iteration: {iteration} passed, feature: {feature} passed")
    # write_json_lists_to_file(f'results/{iteration}/prompts_utility_model_{iteration}_{feature}.jsonl', prompts_to_write)
    write_json_lists_to_file(f'results/rejection3/prompts_similarity_model_{feature}.jsonl', prompts_to_write)
    
    model = get_model(cfg.gen_model)
    
    results = model.predict_multi(prompts, max_workers=4)
    
    results_temp = []
    for res in results:
        results_temp.append(res[1])
    
    for i,comment in enumerate(original_comments): # CHANGE HERE
        comment["similarity_response"] = results_temp[i]
        
        
    # filename = f"results/{iteration}/feedback_utility_{iteration}_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    filename = f"results/rejection3/similarity_utility_{feature}_{''.join(map(str, hardness))}_{cfg.gen_model.name.split('/')[1]}.jsonl"
    write_json_lists_to_file(filename, original_comments)