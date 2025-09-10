from data.reddit import (create_baseline_prompt,create_sanitization_prompt,
                         load_data, load_json_obj_from_file, load_plain_json,
                         write_plain_json,
                         write_json_lists_to_file,create_topic_prior_prompt,
                         create_topic_posterior_prompt, create_sentence_similarity_prompt, create_feedback_inference_prompt,
                         create_self_verification_sanitization_prompt, create_truth_confidence_score_inference_prompt, create_topic_prompt, read_json,
                         create_topic_prior_guess_confidence_prompt, 
                         create_self_verification_sanitization_deception_prompt, create_explicit_sanitization_prompt)
from src.configs import Config, Experiment
from data.filter import filter_profiles, get_unique_private_attribute, get_topics_for_features
from src.reddit.reddit_utils import process_sanitized_response, extract_text_inside_an_tag, extract_comments_after_hash,get_real_value_for_user, extract_topics_from_response
from src.models.model_factory import get_model


def run_synthetic_final(cfg: Config, experiment, feature, hardness) -> None:
    if experiment == Experiment.EXPERIMENT_INFERENCE.value:
        run_inference_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_SANITIZATION.value:
        run_sanitization_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_EXPLICIT_SANITIZATION.value:
        run_explicit_sanitization_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_POST_INFERENCE.value:
        run_post_sanitization_confidence(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_POST_INFERENCE_TRUTH_CONFIDENCE.value:
        run_post_sanitization_confidence(cfg, feature, hardness, only_guess=False)
    if experiment == Experiment.EXPERIMENT_LLM_UTILITY.value:
        run_llm_utility_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_TOPIC.value:
        run_get_conversation_topic_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_TOPIC_PRIOR.value:
        run_get_conversation_topic_prior_experiment(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_DECEPTION_POST_SANITIZATION_CONFIDENCE.value:
        run_deception_post_sanitization_confidence(cfg, feature, hardness)
    if experiment == Experiment.EXPERIMENT_SANITIZATION_DECEPTION.value:
        run_sanitization_deception_experiment(cfg, feature, hardness)
        
def get_private_values():
    features = ['income_level', 'age', 'sex', 'relationship_status']
    map_feat_private_values = {}
    hardness = [1,2,3,4,5]
    for feature in features:
        filename = f"results/rejection3/sanitization_{feature}_{''.join(map(str, hardness))}_Meta-Llama-3.1-70B-Instruct.jsonl"
        comments = load_json_obj_from_file(filename)
    
        private_values = get_unique_private_attribute(comments, feature)
        map_feat_private_values[feature] = private_values
    return map_feat_private_values

def get_id_question_map():
    raw_json = "data/synthpai.jsonl"
    all_conv = read_json(raw_json)
    question_id_question_map = {}
    for conv in all_conv:
        question_id = conv['id']
        question = conv['text']
        question_id_question_map[question_id] = question
        assert question is not None or question != ''
    return question_id_question_map
    
    
def run_inference_experiment(cfg, feature, hardness):
    
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')
    
    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')
    
    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    combined_pre_sanitization_prompts = {}
    map_feat_private_values = get_private_values()

    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        combined_pre_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            combined_pre_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)
            combined_pre_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            for feature in leaked_features:
                # real_value = get_real_value_for_user(map_question_user_answers, question_id, username, feature)
                
                temp_comment = {'username': username, 'text': all_comments}
                prompt = create_feedback_inference_prompt(temp_comment, (feature, map_feat_private_values[feature]))
                combined_pre_sanitization_prompts[question_id][username]['prompts'].append(str(prompt[0]))
    
    for question_id, user_comments in combined_pre_sanitization_prompts.items():
        for username, comments in user_comments.items():
            results = []
            for prompt in comments['prompts']:
                res = model(str(prompt))
                print('------------------------')
                print(str(prompt))
                print('********'*10)
                print(res)
                results.append(res)
                print('------------------------')
            combined_pre_sanitization_prompts[question_id][username]['results'] = results[:]
    filename = f"results/finalmarch8/pre_sanitization_guess_{cfg.gen_model.name}.jsonl"
    write_plain_json(combined_pre_sanitization_prompts, filename)
    
def run_sanitization_experiment(cfg, feature, hardness):
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')

    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')

    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    map_combined_sanitization_prompts = {}
    map_feat_private_values = get_private_values()
    
    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        map_combined_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            map_combined_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)
            map_combined_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            for feature in leaked_features:
                temp_comment = {'username': username, 'text': all_comments}
                prompt = create_self_verification_sanitization_prompt(temp_comment, feature, cfg.task_config)
                map_combined_sanitization_prompts[question_id][username]['prompts'].append(str(prompt[0]))
    t = 0
    for question_id, user_comments in map_combined_sanitization_prompts.items():
        t += 1
        # if t > 2:
        #     continue
        for username, comments in user_comments.items():
            results = []
            tokens = []
            for prompt in comments['prompts']:
                res, usage = model.call_with_metadata(str(prompt))
                print('-------'*10)
                print(res)
                print('-------'*10)
                results.append(res)
                tokens.append((usage.prompt_tokens, usage.completion_tokens, usage.total_tokens))
            map_combined_sanitization_prompts[question_id][username]['sanitization_results'] = results[:]
            map_combined_sanitization_prompts[question_id][username]['tokens'] = tokens[:]
    filename = f"results/finalmarch8aug8/self_iterative_combined_sanitization_{cfg.gen_model.name}.jsonl"
    write_plain_json(map_combined_sanitization_prompts, filename)    

def perform_operation(response, feature, all_comments_for_this_user, model, username, real_value, only_guess=True):
    """response is the sanitization text for the feature a"""
    """other_features is the list of features other than a"""
    map_feat_private_values = get_private_values()
    results = []
    """extract the response"""
    # sanitized_comments = extract_text_inside_an_tag(response)
    
    # for explicit
    sanitized_comments = extract_comments_after_hash(response)
    if len(sanitized_comments) == 0 or abs(len(sanitized_comments) - len(all_comments_for_this_user)) > 1:
        if not (len(sanitized_comments) == 1 and len(all_comments_for_this_user) > 1):
            print('wild ',len(sanitized_comments), len(all_comments_for_this_user))
            for a in all_comments_for_this_user:
                print(f'{a.get("text")}\n')
            print('gpppp')
            for a in sanitized_comments:
                print(f'{a}\n')
            assert len(sanitized_comments) != 0 and abs(len(sanitized_comments) - len(all_comments_for_this_user)) <= 1
    
    """prompt to predict a feature given response for all feature in other_features"""
    temp_comment = {'username': username, 'text': '\n\n'.join(sanitized_comments)}
    if only_guess:
        prompt = create_feedback_inference_prompt(temp_comment, (feature, map_feat_private_values[feature]))
    else:
        pass
        # prompt = create_truth_confidence_score_inference_prompt(temp_comment, (feature, map_feat_private_values[feature]), real_value)
    res, usage = model.call_with_metadata(str(prompt[0]))
    print(res)
    print('************'*10)
    results.append({
        'feature': feature,
        'response': res,
        'post_sanitization_tokens': (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens),
        'prompt': str(prompt[0])
    })
    return results


def run_post_sanitization_confidence(cfg, feature, hardness, only_guess=True):
    # san_filename = 'results/finalmarch8/combined_sanitization_llama3.jsonl'
    # san_filename = 'results/finalmarch8/combined_sanitization_deception_llama3.jsonl'
    # san_filename = 'results/finalmarch8/combined_sanitization_deception_deepseek32b.jsonl'
    # san_filename = 'results/finalmarch8/combined_sanitization_gpt4_1.jsonl'
    sanitization_step = 4
    san_filename = f'results/finalmarch8aug8/{sanitization_step}_combined_sanitization_gpt4_1.jsonl'
    
    sanitized_map_q_a = load_plain_json(san_filename)
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')
    model = get_model(cfg.gen_model)
    t = 0
    for question_id, user_comments in sanitized_map_q_a.items():
        t += 1
        # if t > 2:
        #     continue
        
        
        for username, sanitized_comments in user_comments.items():
            """sanitized_comments is of form {prompts: [], features: [], sanitization_results: []}"""
            """Make sure len of features is same as len of sanitization_results"""
            assert len(sanitized_comments['features']) == len(sanitized_comments['sanitization_results'])
            
            features = sanitized_comments['features']
            sanitization_results = sanitized_comments['sanitization_results']            

            all_comments_for_this_user = map_question_user_answers[question_id][username]
            sanitized_map_q_a[question_id][username]['post_sanitization'] = {}
            
            for i, (feature, result) in enumerate(zip(features, sanitization_results)):
                print(question_id, username, feature, t)             
                real_value = None#get_real_value_for_user(map_question_user_answers, question_id, username, feature)  
                combined_result = perform_operation(result, feature, all_comments_for_this_user, model, username, real_value, only_guess)
                
                sanitized_map_q_a[question_id][username]['post_sanitization'][feature] = combined_result
    if only_guess:
        filename = f"results/finalmarch8aug8/{sanitization_step}_post_sanitization_guess_{cfg.gen_model.name}_by_gpt4_1.jsonl"
    # else:
    #     filename = f"results/finalmarch8/post_sanitization_truth_confidence_{cfg.gen_model.name}_by_gpt4_1.jsonl"
    write_plain_json(sanitized_map_q_a, filename)   
    
def run_llm_utility_experiment(cfg, feature, hardness):
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')

    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')
    
    
    # san_filename = 'results/finalmarch8/combined_sanitization_deception_deepseek32b.jsonl'
    # san_filename = 'results/finalmarch8/combined_sanitization_deepseek32b.jsonl'
    san_filename = 'results/finalmarch8aug8/4_combined_sanitization_gpt4_1.jsonl'
    sanitized_map_q_a = load_plain_json(san_filename)

    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    map_combined_sanitization_prompts = {}
    
    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        map_combined_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            map_combined_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)
            map_combined_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            sanitized_comments = sanitized_map_q_a[question_id][username]['sanitization_results']
            for i, feature in enumerate(leaked_features):
                print(question_id, username, feature)
                assert feature == sanitized_map_q_a[question_id][username]['features'][i]
                # temp_comment = {'username': username, 'text': all_comments}
                
                # sanitized_comments_per_feature = extract_text_inside_an_tag(sanitized_comments[i])
                sanitized_comments_per_feature = extract_comments_after_hash(sanitized_comments[i])
                
                
                
                sanitized_comments_per_feature = '\n\n'.join(sanitized_comments_per_feature)
                prompt = create_sentence_similarity_prompt(all_comments, sanitized_comments_per_feature)
                # prompt = create_self_verification_sanitization_prompt(temp_comment, feature, cfg.task_config)
                map_combined_sanitization_prompts[question_id][username]['prompts'].append(str(prompt))
    t = 0
    for question_id, user_comments in map_combined_sanitization_prompts.items():
        t += 1
        for username, comments in user_comments.items():
            print(question_id, username, t)
            results = []
            for prompt in comments['prompts']:
                res = model(str(prompt))
                print('-------'*10)
                print(res)
                print('-------'*10)
                results.append(res)
            map_combined_sanitization_prompts[question_id][username]['llm_similarity'] = results[:]
        filename = f"results/finalmarch8aug8/llm_utility_{cfg.gen_model.name}_by_gpt4_1.jsonl"
        write_plain_json(map_combined_sanitization_prompts, filename)
    

def run_get_conversation_topic_experiment(cfg, feature, hardness):
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')
    
    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')
    

        
    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    combined_pre_sanitization_prompts = {}
    map_feat_private_values = get_private_values()
    question_id_question_map = get_id_question_map()
    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        combined_pre_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            combined_pre_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)
            combined_pre_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            for feature in leaked_features:
                # real_value = get_real_value_for_user(map_question_user_answers, question_id, username, feature)
                
                temp_comment = {'username': username, 'text': all_comments}
                prompt = create_topic_prompt(question_id_question_map[question_id], temp_comment, (feature, map_feat_private_values[feature]))
                combined_pre_sanitization_prompts[question_id][username]['prompts'].append(str(prompt[0]))
    
    for question_id, user_comments in combined_pre_sanitization_prompts.items():
        for username, comments in user_comments.items():
            results = []
            for prompt in comments['prompts']:
                res = model(str(prompt))
                print('------------------------')
                print(str(prompt))
                print('********'*10)
                print(res)
                results.append(res)
                print('------------------------')
            combined_pre_sanitization_prompts[question_id][username]['results'] = results[:]
    filename = f"results/finalmarch8/topic_guess_{cfg.gen_model.name}.jsonl"
    write_plain_json(combined_pre_sanitization_prompts, filename)

def perform_topic_prior_operation(response, feature, all_comments_for_this_user, model, username):
    """response is the sanitization text for the feature a"""
    """other_features is the list of features other than a"""
    map_feat_private_values = get_private_values()
    results = []
    """extract the response"""
    _, topics = extract_topics_from_response(response, None)
    assert topics is not None or topics != '' or len(topics) != 0
    """prompt to predict a feature given response for all feature in other_features"""
    temp_comment = {'username': username, 'text': topics}
    prompt = create_topic_prior_guess_confidence_prompt(temp_comment, (feature, map_feat_private_values[feature]))
    res =  model(str(prompt[0]))
    print(res)
    print('************'*10)
    results.append({
        'feature': feature,
        'response': res,
        'prompt': str(prompt[0])
    })
    return results

def run_get_conversation_topic_prior_experiment(cfg, feature, hardness):
    
    qid_cluster_map = load_plain_json(cfg.task_config.path+'questionid_cluster.json')
    cluster_topic_map = load_plain_json(cfg.task_config.path+'cluster_topics.json')
    map_feat_private_values = get_private_values()
    feature_with_labels = {
        'income_level': ['low', 'middle', 'high', 'very high'],
        'sex': ['male', 'female'],
        'age': ['1-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100'],
        'relationship_status': ['single', 'in relationship', 'married', 'divorced', 'widowed', 'engaged'],
    }
    
    cluster_feature_response = {}
    model = get_model(cfg.gen_model)
    for cluster_id, topics in cluster_topic_map.items():
        cluster_feature_response[cluster_id] = {}
        for feat in feature_with_labels.keys():
            cluster_feature_response[cluster_id][feat] = {}
            prompt = create_topic_prior_guess_confidence_prompt(topics, (feat, map_feat_private_values[feat]))
            cluster_feature_response[cluster_id][feat]['prompt'] = str(prompt[0])
            res = model(str(prompt[0]))
            print(res)
            print('************'*10)
            cluster_feature_response[cluster_id][feat]['response'] = res
            
    filename = f"results/finalmarch8/topic_prior_{cfg.gen_model.name}.jsonl"
    write_plain_json(cluster_feature_response, filename)          

def perform_deception_operation(response, other_features, all_comments_for_this_user, model, username):
    """response is the sanitization text for the feature a"""
    """other_features is the list of features other than a"""
    map_feat_private_values = get_private_values()
    results = []
    """extract the response"""
    sanitized_comments = extract_text_inside_an_tag(response)
    if len(sanitized_comments) != len(all_comments_for_this_user):
        print('wild')
        for a in all_comments_for_this_user:
                print(f'{a.get("text")}\n')
        print('gpppp')
        for a in sanitized_comments:
            print(f'{a}\n')
    assert len(sanitized_comments) == len(all_comments_for_this_user)
    
    """prompt to predict a feature given response for all feature in other_features"""
    for other_feature in other_features:
        temp_comment = {'username': username, 'text': '\n\n'.join(sanitized_comments)}
        prompt = create_feedback_inference_prompt(temp_comment, (other_feature, map_feat_private_values[other_feature]))
        res = model(str(prompt[0]))
        print(res)
        print('************'*10)
        results.append({
            'feature': other_feature,
            'response': res,
            'prompt': str(prompt[0])
        })
    return results

"""Dont use this. not required"""
def run_deception_post_sanitization_confidence(cfg, feature, hardness):
    # san_filename = 'results/finalmarch8/combined_sanitization_llama3.jsonl'
    san_filename = 'results/finalmarch8/combined_sanitization_deception_deepseek32b.jsonl'
    
    sanitized_map_q_a = load_plain_json(san_filename)
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')
    model = get_model(cfg.gen_model)
    for question_id, user_comments in sanitized_map_q_a.items():
        for username, sanitized_comments in user_comments.items():
            """sanitized_comments is of form {prompts: [], features: [], sanitization_results: []}"""
            """Make sure len of features is same as len of sanitization_results"""
            assert len(sanitized_comments['features']) == len(sanitized_comments['sanitization_results'])
            
            features = sanitized_comments['features']
            sanitization_results = sanitized_comments['sanitization_results']            

            all_comments_for_this_user = map_question_user_answers[question_id][username]
            sanitized_map_q_a[question_id][username]['post_sanitization'] = {}
            
            for i, (feature, result) in enumerate(zip(features, sanitization_results)):
                other_features = [f for j, f in enumerate(features) if j != i] 
                # if question_id == 'QtENSXFf39' and username == 'XylophoneXenon':
                print(question_id, username, feature)           
                combined_result = perform_deception_operation(result, other_features, all_comments_for_this_user, model, username)
                sanitized_map_q_a[question_id][username]['post_sanitization'][feature] = combined_result
    filename = f"results/finalmarch8/deception_post_sanitization_confidence_{cfg.gen_model.name}_by_deepseek32b.jsonl"
    write_plain_json(sanitized_map_q_a, filename)   
    
    
def run_sanitization_deception_experiment(cfg, feature, hardness):
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')

    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')

    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    map_combined_sanitization_prompts = {}
    map_feat_private_values = get_private_values()
    
    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        map_combined_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            map_combined_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)
            map_combined_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            for feature in leaked_features:
                real_value = get_real_value_for_user(map_question_user_answers, question_id, username, feature)
                temp_comment = {'username': username, 'text': all_comments}
                prompt = create_self_verification_sanitization_deception_prompt(temp_comment, feature, cfg.task_config, real_value)
                map_combined_sanitization_prompts[question_id][username]['prompts'].append(str(prompt[0]))
    
    for question_id, user_comments in map_combined_sanitization_prompts.items():
        for username, comments in user_comments.items():
            results = []
            for prompt in comments['prompts']:
                res = model(str(prompt))
                print('-------'*10)
                print(res)
                print('-------'*10)
                results.append(res)
            map_combined_sanitization_prompts[question_id][username]['sanitization_results'] = results[:]
    filename = f"results/finalmarch8/combined_sanitization_deception_{cfg.gen_model.name}.jsonl"
    write_plain_json(map_combined_sanitization_prompts, filename)      
    
    
    
def run_explicit_sanitization_experiment(cfg, feature, hardness):
    sanitization_step = 4
    map_question_user_answers = load_plain_json(cfg.task_config.path+'q_u_comments_whole.json')

    map_question_user_features = load_plain_json(cfg.task_config.path+'q_u_leaked_features_whole.json')

    model = get_model(cfg.gen_model)
    print('Model loaded')
    
    map_combined_sanitization_prompts = {}
    map_feat_private_values = get_private_values()
    
    if sanitization_step > 1:
        sanitized_file = f'results/finalmarch8aug8/{sanitization_step-1}_combined_sanitization_gpt4_1.jsonl'
        post_san_inference_file = f'results/finalmarch8aug8/{sanitization_step-1}_post_sanitization_guess_gpt4_1_by_gpt4_1.jsonl'

        sanitized_prev_map_q_a = load_plain_json(sanitized_file)
        inference_prev_map_q_a = load_plain_json(post_san_inference_file)
        
        
    for question_id, user_comments in map_question_user_answers.items():
        if not user_comments: continue
        map_combined_sanitization_prompts[question_id] = {}
        for username, comments in user_comments.items():
            if not comments: continue
            map_combined_sanitization_prompts[question_id][username] = {'prompts': []}
            leaked_features = map_question_user_features[question_id][username]
            all_comments = [comment['text'].strip()+'\n\n' for comment in comments]
            all_comments = ''.join(all_comments)                
            map_combined_sanitization_prompts[question_id][username] = {'prompts': [], 'features': leaked_features}
            all_comments_for_this_user = map_question_user_answers[question_id][username]
            
            for feature in leaked_features:
                cues = None
                if sanitization_step > 1:
                    sanitization_results = sanitized_prev_map_q_a[question_id][username]['sanitization_results']
                    feature_index = sanitized_prev_map_q_a[question_id][username]['features'].index(feature)
                    all_comments = sanitization_results[feature_index]
                    all_comments = extract_comments_after_hash(all_comments)
                    
                    if len(all_comments) == 0 or abs(len(all_comments) - len(all_comments_for_this_user)) > 1:
                        if not (len(all_comments) == 1 and len(all_comments_for_this_user) > 1):
                            print('wild ',len(all_comments), len(all_comments_for_this_user))
                        assert len(all_comments) != 0 and abs(len(all_comments) - len(all_comments_for_this_user)) <= 1, print(len(all_comments))
                    all_comments = ''.join([ac.strip()+'\n\n' for ac in all_comments])       
                    print(question_id, username, feature)
                    cues = inference_prev_map_q_a[question_id][username]['post_sanitization'][feature][0]['response']
                    if isinstance(cues, str):
                        import json
                        cues = json.loads(cues)
                        cues = cues['Inference']
                    else:
                        cues = cues['Inference']
                    assert cues is not None or cues != ''
                    
                temp_comment = {'username': username, 'text': all_comments}
                prompt = create_explicit_sanitization_prompt(temp_comment, feature, cfg.task_config, cues=cues)
                map_combined_sanitization_prompts[question_id][username]['prompts'].append(str(prompt[0]))
    
    counter = 0
    for question_id, user_comments in map_combined_sanitization_prompts.items():
        # if counter == 2:
        #     break
        for username, comments in user_comments.items():
            results = []
            tokens = []
            for prompt in comments['prompts']:
                res, usage = model.call_with_metadata(str(prompt))
                print('-------'*10)
                print(res)
                print('-------'*10)
                results.append(res)
                # results.append(res.choices[0].message.content)
                tokens.append((usage.prompt_tokens, usage.completion_tokens, usage.total_tokens))
            map_combined_sanitization_prompts[question_id][username]['sanitization_results'] = results[:]
            map_combined_sanitization_prompts[question_id][username]['tokens'] = tokens[:]
        counter += 1
    filename = f"results/finalmarch8aug8/{sanitization_step}_combined_sanitization_{cfg.gen_model.name}.jsonl"
    write_plain_json(map_combined_sanitization_prompts, filename) 