import json
import re
from datasets import load_dataset
import math


def type_to_str(pii_type: str) -> str:
    if pii_type == "income_level":
        pii_type_str = "yearly income"
    elif pii_type == "education":
        pii_type_str = "level of education"
    elif pii_type == "pobp":
        pii_type_str = "place of birth"
    elif pii_type == "location":
        pii_type_str = "current place of living"
    elif pii_type == "relationship_status":
        pii_type_str = "relationship status"
    elif pii_type == 'sex':
        pii_type_str = 'gender'
    else:
        pii_type_str = pii_type

    return pii_type_str

def type_to_options(pii_type: str) -> str:
    if pii_type == "income_level":
        pii_type_str = " Choose from these options: (low, middle, high, very high); low if <30k USD, middle if 30-60k USD, high if 60-150k USD, very high if >150k USD."
        # pii_type_str = " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)."
    elif pii_type == "education":
        pii_type_str = " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD."
    elif pii_type == "sex":
        pii_type_str = " Choose from these options: Male, Female."
    elif pii_type == "relationship_status":
        pii_type_str = (
            # " Choose from these options: No relation, In Relation, Married, Divorced."
            " Choose from these options: Widowed, Single, Married, Engaged, In relationship, Divorced"
        )
    elif pii_type == "age":
        # pii_type_str = " Use the age of the author when he wrote the comment if date is available."
        pii_type_str = " Please provide an age range with a window size of 10 years. For example: 34-44"
    else:
        pii_type_str = ""

    return pii_type_str
    
def process_sanitized_response(sanitized_response):
    responses = []
    for response in sanitized_response.split("\n"):
        if response == "" or response == " ":
            continue
        responses.append(response)
    assert responses[-1] is not None
    return responses[-1]

def extract_inference_from_response(response, id):
    if isinstance(response, dict):
        # print(response)
        response = {key.lower(): value for key, value in response.items()}
        if 'inference' in response and 'guess' in response and 'confidence' in response:
            return response['inference'], response['guess'], response['confidence']
    # print(id)
    # Extract the JSON part inside the response using regex
    json_match = re.search(r'```(.*?)```', response, re.DOTALL)
        # Convert bytes to string if needed
    if isinstance(response, bytes):
        response = response.decode("utf-8")
    
    # Unescape escaped sequences
    response = response.encode('utf-8').decode('unicode_escape')

    # Try to find JSON block wrapped in triple backticks
    json_match = re.search(r'```json(.*?)```', response, re.DOTALL)
    if not json_match:
        json_match = re.search(r'```(.*?)```', response, re.DOTALL)
        
        
    if json_match:
        json_str = json_match.group(1).strip()  # Extract the JSON part
    else:
        json_match = re.search(r'\{[\s\S]*?\}', response, re.DOTALL)
        if not json_match:
            print('Error loading json')
            raise ValueError(f"Problem loading json: {id}")
        json_str = json_match.group(0).strip()        
    try:
        json_data = json.loads(json_str)  # Parse JSON
    except:
        print('Error loading json')
        raise ValueError(f"Problem loading json: {id}")
    
    # Normalize key search: Convert all keys to lowercase
    inference_key = next((key for key in json_data if key.lower() == "inference"), None)
    guess_key = next((key for key in json_data if key.lower() == "guess"), None)
    confidence_key = next((key for key in json_data if key.lower() == "confidence"), None)
    assert inference_key is not None and guess_key is not None
    assert json_data[inference_key] is not None and json_data[guess_key] is not None and json_data[confidence_key] is not None
    assert json_data[inference_key] != "" and json_data[guess_key] != "" and json_data[confidence_key] != ""
    return json_data[inference_key], json_data[guess_key], json_data[confidence_key]

def extract_anonymized_txt_from_response(response, id=None):
    # Extract the comment after #
    match = re.search(r'#\s*(.*)', response)

    if match:
        comment = match.group(1).strip()
        return comment
    else:
        print("No comment found.")
        raise ValueError(f'No response or anonymizedzed text for id {id}')
    
    
def load_sts_dataset():
    dataset = load_dataset('glue', 'stsb')
    validation_dataset = dataset['validation']
    sentence_pairs = [(id, label, t1, t2) for id, label, t1,t2 in zip(validation_dataset['idx'],validation_dataset['label'],validation_dataset['sentence1'], validation_dataset['sentence2'])]
    human_scores = [score['label'] for score in validation_dataset]
    return sentence_pairs, human_scores

def extract_similarity_scores(response, id=None):
    if isinstance(response, dict):
        # print(response)
        response = {key.lower(): value for key, value in response.items()}
        if 'readability' in response and 'meaning' in response:
            return response['readability']['score'], response['meaning']['score']
    response = response.encode('utf-8').decode('unicode_escape')
    if '</think>' in response:
        response = response.split('</think>')[-1]
    response = re.sub(r"```(?:json)?", "", response).strip()
    
    json_match =  re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        print('Error loading json ')
        raise ValueError(f"Problem loading json: {id} :: {response}")
    json_str = json_match.group(0).strip()  
    if not json_str:
        print(f"Locha in this {id}")
    try:
        json_data = json.loads(json_str)  # Parse JSON
    except:
        print('Error loading json ', id)
        raise ValueError(f"Problem loading json: {json_str} :: {type(json_str)}")
    
    # Normalize key search: Convert all keys to lowercase
    readibility_key = next((key for key in json_data if key.lower() == "readability"), None)
    
    meaning_key = next((key for key in json_data if key.lower() == "meaning"), None)
    
    assert readibility_key is not None and meaning_key is not None
    assert json_data[readibility_key] is not None and json_data[meaning_key] is not None
    assert json_data[readibility_key]['score'] is not None and json_data[meaning_key]['score'] is not None
    if not isinstance(json_data[readibility_key]['score'],  (int, float)) or not isinstance(json_data[meaning_key]['score'],  (int, float)):
        print('Error loading json ')
        raise ValueError(f"Problem loading json: {response}: id {id}")
    return json_data[readibility_key]['score'], json_data[meaning_key]['score']

def extract_last_anonymized_text(text):
    # Split into sections based on `#`
    sections = text.strip().split("\n#")

    # Get the last section after the last `#`
    last_section = sections[-1].strip()
    # print(f'This is last: {last_section}')
    # print(f'End last')
    
    first_line = last_section.split('\n')[0]
    if first_line.strip()[-1] == ':':
        return last_section.split('\n')[1]
    elif ':' in first_line.strip():
        return first_line[first_line.index(':')+1:].strip()
    if not ':' in first_line.strip():
        return first_line.strip()
    
    # Extract the part after ":" if it exists, otherwise take the whole line
    match = re.search(r":\s*(.+)", last_section)
    final_match = match.group(1) if match else last_section
    assert final_match is not None or final_match != ""
    return final_match
    
import re

def extract_text_inside_an_tag(text):
    # Use regex to find content inside <an> </an> tags
    # match = re.search(r'<an>(.*?)</an>', text, re.DOTALL)
    matches = re.findall(r'<an>(.*?)</an>', text, re.DOTALL)

    # Get the last match if any exist
    match = matches[-1] if matches else None
    
    if match:
        # Extract the content inside the <an> tag
        # extracted_text = match.group(1).strip()
        extracted_text = match.strip()
        
        # Split into sentences based on new lines
        sentences = [sentence.strip() for sentence in extracted_text.split('\n') if sentence.strip()]
        
        return sentences
    else:
        return []
    
def extract_comments_after_hash(text):

    comments = []
    # Split the input string into individual lines.
    lines = text.splitlines()  
    found = False
    for line in lines:
        # Check if the line starts with '#' after stripping leading/trailing whitespace.
        if line.strip().startswith('#'):
            found = True
            # Remove the '#' and any following whitespace, then add to the comments list.
            # comments.append(line.strip().lstrip('#').strip())  
            continue
        if found:
            if line.strip():
                comments.append(line.strip())
    return comments

def number_to_range_string(number):
    # start = ((number - 1) // 10) * 10 + 1
    # end = start + 9
    start = number - 5
    end = number + 5
    return f"{start}-{end}"

def parse_to_int(s):
    if '-' in s:
        # Split the string into two parts, start and end of the range
        start, end = s.split('-')
        # Convert the start and end to integers
        start = int(start)
        end = int(end)
        # Calculate the average and return the floor value
        average = (start + end) / 2
        return math.floor(average)
    else:
        return int(s)
    
def process_estimate(estimate, feature):
    return estimate
    if feature == 'age':
        estimate = parse_to_int(estimate)        
        return estimate#number_to_range_string(estimate)
    return estimate

def get_real_value_for_user(map_question_user_answers, question_id, username, feature) -> list:
    """Given map of question: [users]: [comments], get the real value for the feature"""
    
    """Get the comments from the map_question_user_answers"""
    user_comments = map_question_user_answers[question_id][username]
    labels_for_feature = set()
    
    for comment in user_comments:
        """Get the real feature value from human review"""
        if comment['reviews']['human'][feature]['estimate'] != '':
            label = process_estimate(comment['reviews']['human'][feature]['estimate'], feature)
            labels_for_feature.add(label.lower())
    # if feature == 'sex':
        # print(labels_for_feature)
    """Make sure there's at least one single value for this feature"""
    assert len(labels_for_feature) > 0
    # """Make sure that thaere's no multiple values"""
    # assert len(labels_for_feature) == 1  
    
    """return the value"""
    return list(labels_for_feature)[0]

def extract_topics_from_response(response, id):
    if isinstance(response, dict):
        # print(response)
        response = {key.lower(): value for key, value in response.items()}
        if 'topics' in response and 'inference' in response:
            return response['inference'], response['topics']
    # print(id)
    # Extract the JSON part inside the response using regex
    json_match = re.search(r'```(.*?)```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()  # Extract the JSON part
    else:
        json_match = re.search(r'\{[\s\S]*?\}', response, re.DOTALL)
        if not json_match:
            print('Error loading json')
            raise ValueError(f"Problem loading json: {id}")
        json_str = json_match.group(0).strip()        
    try:
        json_data = json.loads(json_str)  # Parse JSON
    except:
        print('Error loading json')
        raise ValueError(f"Problem loading json: {id}")
    
    # Normalize key search: Convert all keys to lowercase
    inference_key = next((key for key in json_data if key.lower() == "inference"), None)
    topics_key = next((key for key in json_data if key.lower() == "topics"), None)
    assert inference_key is not None and topics_key is not None
    assert json_data[inference_key] is not None and json_data[topics_key] is not None
    return json_data[inference_key], json_data[topics_key]