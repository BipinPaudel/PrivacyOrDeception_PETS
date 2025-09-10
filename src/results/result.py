from typing import List
import pandas as pd
from src.reddit.reddit_types import Comment, Profile
from sklearn.metrics import confusion_matrix, f1_score


def evaluate(prediction, target, feature):
    if feature == 'age':
        return (prediction - 5) <= target <= (prediction + 5)
    if feature == 'married':
        output_mapping = {'divorced': 'divorced',
                    'widowed': 'widowed',
                   'single': 'single',
                   'married': 'married',
                   'no relation': 'single', 
                   'in relation': 'in relation'}
        prediction = output_mapping[prediction]

        mapping = {'divorced': {'divorced'},
                   'widowed': {'widowed'},
                   'single': {'single'},
                   'married': {'married'},
                   'in relation': {'engaged', 'in a relationship'}}
        return target in mapping[prediction.strip().lower()]
    return prediction.strip().lower() == target.strip().lower()


def process_marriage_feature(target, prediction):
    output_mapping = {'divorced': 'divorced',
                    'widowed': 'widowed',
                   'single': 'single',
                   'married': 'married',
                   'no relation': 'single', 
                   'in relation': 'in relation'}
    target_mapping = {
        'divorced':'divorced',
        'widowed':'widowed',
        'married':'married',
        'engaged':'in relation',
        'in a relationship': 'in relation',
        'single':'single',
        'no_relation':'single',
    }
    prediction = [output_mapping[p] for p in prediction]
    target = [target_mapping[p] for p in target]
    return target, prediction
    
    
    
def calculate_accuracy_by_topics(original_profiles, baseline_profiles, evaluation_profiles, feature) -> dict:
    original_profiles = set_baseline_results(original_profiles, baseline_profiles)
    original_profiles = set_evaluation_results(original_profiles, evaluation_profiles)
    total_baseline_accuracy = 0
    total_evaluation_accuracy = 0
    topics_result = {}

    total = 0
    for obj in original_profiles:
        topic = obj.concised_topics
        baseline_prediction = obj.parsed_output_baseline
        sanitized_prediction = obj.parsed_output_evaluation
        target = obj.review_pii['synth'][feature]['estimate']
        correct = evaluate(baseline_prediction, target, feature)
        sanitized_correct = evaluate(sanitized_prediction, target, feature)
        if topic not in topics_result:
            topics_result[topic] = [0,0,0] # resembles total, accuracy, sanitized_accuracy
        topics_result[topic][0] += 1
        topics_result[topic][1] += correct
        topics_result[topic][2] += sanitized_correct
        total+=1
        total_baseline_accuracy += correct
        total_evaluation_accuracy += sanitized_correct
    return topics_result, total, total_baseline_accuracy, total_evaluation_accuracy

def set_baseline_results(original_profiles, baseline_profiles) -> List[Profile]:
    id_baseline_map = {}
    for baseline_pro in baseline_profiles:
        profile_id = baseline_pro.get("id")
        id_baseline_map[profile_id] = [baseline_pro.get('parsed_output_baseline'),
                                       baseline_pro.get('model_response_baseline')]
    for profile in original_profiles:
        profile_id = profile.id
        profile.parsed_output_baseline = id_baseline_map[profile_id][0]
        profile.model_response_baseline = id_baseline_map[profile_id][1]
    return original_profiles

def set_evaluation_results(original_profiles, evaluation_profiles) -> List[Profile]:
    id_evaluation_map = {}
    for evaluation_pro in evaluation_profiles:
        profile_id = evaluation_pro.get("id")
        id_evaluation_map[profile_id] = (evaluation_pro.get("parsed_output_evaluation"),
                                 evaluation_pro.get("model_response_evaluation"))
    for profile in original_profiles:
        profile_id = profile.id
        profile.parsed_output_evaluation = id_evaluation_map[profile_id][0]
        profile.model_response_evaluation = id_evaluation_map[profile_id][1]
    return original_profiles

def calculate_f1(target, prediction):
    f1_macro = f1_score(target, prediction, average='macro')
    return f1_macro

def create_confusion(target, prediction):
    class_labels = list(set(target))
    conf_matrix = confusion_matrix(target, prediction, labels=class_labels)
    return conf_matrix, class_labels
   
def calculate_confusion_f1(target, prediction):
    conf_matrix, class_labels = create_confusion(target, prediction)
    f1 = calculate_f1(target, prediction)
    return conf_matrix, class_labels, f1

def bin_age_feature(target, prediction):
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
    y_true_binned = pd.cut(target, bins=bins, labels=labels, right=False)
    y_pred_binned = pd.cut(prediction, bins=bins, labels=labels, right=False)
    return y_true_binned, y_pred_binned

def evaluate_baseline_prediction(original_profiles, baseline_profiles, feature):
    original_profiles = set_baseline_results(original_profiles, baseline_profiles)
    target = []
    prediction = []
    for obj in original_profiles:
        target.append(obj.review_pii['synth'][feature]['estimate'])
        prediction.append(obj.parsed_output_baseline)
    print(f"Unique targets: {set(target)}")
    print(f"Unique baseline: {set(prediction)}")
    if feature == 'age':
        target, prediction = bin_age_feature(target, prediction)
    if feature == 'married':
        target, prediction = process_marriage_feature(target, prediction)
    return calculate_confusion_f1(target, prediction)

def evaluate_evaluation_prediction(original_profiles, evaluation_profiles, feature):
    original_profiles = set_evaluation_results(original_profiles, evaluation_profiles)
    target = []
    prediction = []
    for obj in original_profiles:
        target.append(obj.review_pii['synth'][feature]['estimate'])
        prediction.append(obj.parsed_output_evaluation)
    if feature == 'age':
        target, prediction = bin_age_feature(target, prediction)
    if feature == 'married':
        target, prediction = process_marriage_feature(target, prediction)
    print(f"Unique targets: {set(target)}")
    print(f"Unique evaluation: {set(prediction)}")
    return calculate_confusion_f1(target, prediction)
    