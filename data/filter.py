import re

def filter_profiles(profiles, feature, hardness=[1,2,3,4,5]):
    filtered_profiles = []
    for profile in profiles:
        pii_type_hardness_dict = pii_type_with_hardness_of_a_profile(profile)
        if feature in pii_type_hardness_dict and pii_type_hardness_dict[feature] in hardness:
            filtered_profiles.append(profile)
    return filtered_profiles

def pii_type_with_hardness_of_a_profile(profile):
    pii_type_hardness_dict = {}
    for pii_type, pii_desc in profile.review_pii['synth'].items():
        pii_type_hardness_dict[pii_type] = pii_desc.get('hardness')
    return pii_type_hardness_dict

def get_unique_private_attribute(comments, feature='income'):
    unique_values = set()
    for comment in comments:
        feature_val = comment['reviews']['human'][feature]['estimate']
        assert feature_val is not None
        unique_values.add(feature_val)
    
    if feature == 'age':
        # Generate the list with range strings
        range_list = []

        # Iterate over the initial list in steps of 10
        for i in range(0, 100, 10):
            start = i + 1
            end = i + 10
            range_list.append(f"{start}-{end}")

        return list(set(range_list))
    return list(unique_values)

def get_topics_for_features(comments, feature='income'):
    return list(set([comment['concised_topics'] for comment in comments]))


def extract_topics_tag(res):
    # Use regex to find the content between <topics> and </topics>
    match = re.search(r'<topics>(.*?)</topics>', res)
    topics_content = match.group(1)
    return topics_content

def extract_probabilities(res):
    pattern = r'<value_probability>(.*?)<\/value_probability>'
    matches = re.findall(pattern, res)
    value_prob_map = {}
    for match in matches:
        values_prob = match.split(':')
        value_prob_map[values_prob[0]] =values_prob[1]
    return value_prob_map

'''
returns map of topic and it's associated prior values in json format
'''
def get_topics_prior_values_map_from_gpt(topic_prior_requests, topic_prior_results):
    id_topic_map = {t['custom_id']: extract_topics_tag(t['body']['messages'][1]['content']) for t in topic_prior_requests}
    topic_values_map = {id_topic_map[t['custom_id']]: extract_probabilities(t['response']['body']['choices'][0]['message']['content']) for t in topic_prior_results}
    return topic_values_map
    
    
    
        