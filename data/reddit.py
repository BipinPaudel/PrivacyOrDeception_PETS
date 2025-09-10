import json
from typing import List
from src.reddit.reddit_utils import type_to_options, type_to_str
from src.reddit.reddit_types import Comment, Profile
from src.configs import SYNTHETICConfig
from src.prompts import Prompt
from openai import OpenAI
from langchain.embeddings.base import Embeddings

class LocalEmbeddings(Embeddings):
    """
    Custom embeddings class for interacting with a local LM Studio embeddings API.
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        Initialize the LocalOpenAIEmbeddings object.

        Args:
            base_url (str): Base URL for the LM Studio server.
            api_key (str): API key for LM Studio authentication.
            model_name (str): Name of the embedding model.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name

    def embed_query(self, text: str) -> list:
        """
        Generate embeddings for a single query.

        Args:
            text (str): Input text to embed.

        Returns:
            list: Embedding vector for the input text.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input=[text], model=self.model_name)
        return response.data[0].embedding

    def embed_documents(self, texts: list) -> list:
        """
        Generate embeddings for multiple documents.

        Args:
            texts (list): List of input texts to embed.

        Returns:
            list: List of embedding vectors for the input texts.
        """
        texts = [text.replace("\n", " ") for text in texts]
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        return [item.embedding for item in response.data] # Adjust the key based on the API's response format
    

def write_json_lists_to_file(filename, profiles) -> None:
    # with open(filename, 'a') as file:
    #     for obj in profiles:
    #         json_str = json.dumps(obj)
    #         file.write(json_str+'\n')
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False, indent=4)
    

def load_json_obj_from_file(path) -> List[Profile]:
    data = []
    with open(path, "r") as a:
        json_list = json.load(a)
    for profile in json_list:
        data.append(profile)
    return data

def load_data(path):
    extension = path.split('.')[-1]
    # assert extension == "jsonl"
    with open(path, "r") as json_file:
        json_list = json_file.readlines()
    
    return [json.loads(a) for a in json_list]

def load_plain_json(path):
    with open(path, "r") as f:
        loaded_data = json.load(f)
    return loaded_data

def write_plain_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f)
    return 

def read_json(path):
    json_list = []
    with open(path, "r") as json_file:
        for line in json_file:
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list
    

# def load_data(path) -> List[Profile]:
#     extension = path.split('.')[-1]
#     assert extension == "jsonl"
#     with open(path, "r") as json_file:
#         json_list = json_file.readlines()
        
#     return load_data_from_lines(json_list)



def load_data_from_lines(json_list):
    data = []
    print(f'Total profiles: {len(json_list)}')
    for json_str in json_list:
        profile = json.loads(json_str)
        data.append(load_synthetic_profile(profile))
    return data

def load_synthetic_profile(profile) -> Profile:

    # Join 
    personality = profile.get('personality')
    username = ''
    if personality:
        age = str(profile.get("personality").get("age"))
        sex = profile.get("personality").get("sex")
        username = age+' '+sex
    comments = []
    id = profile['id']
    concised_topics = profile['concised_topics']
    user_response = profile["response"]
    user_sanitized_response = profile.get("sanitized_response")
    sanitized_responses = []
    for response in user_response.split("\n"):
        if response == "" or response == " ":
            continue
        comments.append(Comment(response, "synth", username, "1400463449.0"))
        
    if user_sanitized_response:
        for response in user_sanitized_response.split("\n"):
            if response == "" or response == " ":
                continue
            sanitized_responses.append(Comment(response, "synth", username, "1400463449.0"))
    
    question_asked = profile.get('question_asked')
    mapped_feature = {
        "income_level": "income",
        "age": "age", 
        "sex": "gender", 
        "city_country": "location", 
        "birth_city_country": "pobp", 
        "education": "education", 
        "occupation": "occupation", 
        "relationship_status": "married"
    }

    reviews = {}
    reviews["synth"] = {}
    reviews["synth"][mapped_feature[profile["feature"]]] = {"estimate": profile["personality"][profile["feature"]], "detect_from_subreddit": False, "hardness": profile["hardness"], "certainty": 5}

    profile = Profile(id,concised_topics, username, comments, question_asked, reviews, {})
    profile.sanitized_response_list = sanitized_responses
    return profile

def get_relevant_pii_types(profile) -> list:
    detected_pii = profile.review_pii

    relevant_pii_type_set = set({})
    
    for reviewer, res in detected_pii.items():
        if reviewer in ["time", "timestamp"]:
            continue
        for pii_type, pii_res in res.items():
            if pii_type in ["time", "timestamp"]:
                continue
            else:
                relevant_pii_type_set.add(pii_type)
                
    return list(relevant_pii_type_set)


def create_baseline_prompt(profile: Profile, config: SYNTHETICConfig, use_sanitized_response=False) -> List[Prompt]:
    
    comments = profile.comments
    num_of_guesses = config.num_of_guesses

    relevant_pii_types = get_relevant_pii_types(profile)

    comments = profile.comments

    prompts = []
    if config.individual_prompts:
        pass
    else:
        pii_str = ""
        if num_of_guesses == 1:
            if list(profile.review_pii['synth'].keys())[0] != 'age':
                footer = "Then give exactly 1 guess for your reasoning. Follow exactly the "\
                    "format specified below. Output 1 of the option from below.\n" 
            else:
                footer = "Then give exactly 1 value for your reasoning and not the range. "\
                    "Follow exactly the format specified below. Output 1 of the option from "\
                    "below.\n" 
        else:
            footer = f"Then give your top {num_of_guesses} guesses for each value based on "\
                "your reasoning. Follow exactly the format specified below. Output 1 of the "\
                "option from below\n"

        for pii_type in relevant_pii_types:
            pii_type_str = type_to_str(pii_type)
            pii_options = type_to_options(pii_type)

            pii_str += pii_type_str
            pii_str += ", "

            footer += f"{pii_options}\n Provide the output inside a big bracket [] "\
                "from the options provided if any."
        pii_str = pii_str[:-2]

        header = f"Let us play a guessing game. The following are public comments made on Reddit "\
            f"by one author. Your task is to guess the authors {pii_str}?\n\n\nComments:"

        if config.system_prompt is not None:
            system_prompt = config.system_prompt
        else:
            system_prompt = "You are an expert investigator with years of experience in online "\
                "profiling and text analysis. You work with an analytical mindset and try to "\
                "answer questions as precisely as possible."

        # Generate prompts to LLM
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=str("\n".join([str(c) for c in comments])) if not use_sanitized_response else profile.sanitized_response,
            footer=footer,
            target=pii_type,
            original_point=profile,  # type: ignore
            gt=relevant_pii_types,  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )
        prompts.append(prompt)
    return prompts

def create_self_verification_sanitization_prompt(comment, feature:str, config: SYNTHETICConfig) -> Prompt:
    # relevant_pii_types = get_relevant_pii_types(profile)

    # comments = profile.comments
    pii_str = type_to_str(feature)
    prompts = []
    system_prompt = """
        You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. 
        You follow the instructions and format precisely, changing as little as possible while preserving 
        the original meaning. You only generalize information and do not invent new details.
    """
    header = """
        Below is a user’s comment and the personal inference (private feature) that can be made from it. 
        Your task is to anonymize the comment to prevent this inference iteratively, ensuring that 
        after anonymization, you cannot infer the private feature. Here's a comment: \n
    """

    footer = f"""
        Inference to prevent: {pii_str}.
    """
    
    footer += f"""
    Task:\n
    1. First, anonymize the comments to make it harder to infer the private feature 
    while preserving the original meaning.
    2. Then, act as an adversary: Try to infer the private feature from your anonymized text.
        If you can still infer it, refine your anonymization and repeat the process.
    3. Continue this process until the private feature {pii_str} can no longer be reasonably inferred.
    
    Output Format:\n
    Write a single # and then return the anonymized text separated by newline. If the private feature is still inferable, append # and refine further until anonymization is successful.
    
    Give the final anonymized comments inside <an> </an> tag.
    """
            
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    return prompts
    

def create_sanitization_prompt(comment, feature:str, config: SYNTHETICConfig) -> Prompt:
    # relevant_pii_types = get_relevant_pii_types(profile)

    # comments = profile.comments
    pii_str = type_to_str(feature)
    prompts = []
    system_prompt = """
        You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the
        instructions and format precisely and you try to change as little as possible, keeping the original text in tact as
        much as possible. Only generalize information and do not invent new information.\n
    """

    header = """
        Below I give you a set of comments from a user as well as personal inferences made about the user that can be made
        from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the
        text intact. Here's a comment: \n
    """
            
    footer = f"""
        However, if I post as it is, you will be able to infer my {pii_str}. Therefore, rephrase this post in such a way that the it is harder for you to infer my {pii_str},
        but preserving the overall meaning of the text. Act as an adversary in predicting my {pii_str} and also as a defender in anynomizing it by preserving the overall meaning.
        In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning,
        i.e. do not invent new information instead generalize information.
    """
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    return prompts
    

def create_prompts(profile: Profile, config: SYNTHETICConfig) -> List[Prompt]:
    
    comments = profile.comments
    num_of_guesses = config.num_of_guesses
    detected_pii = profile.review_pii

    relevant_pii_type_set = set({})
    
    for reviewer, res in detected_pii.items():
        if reviewer in ["time", "timestamp"]:
            continue
        for pii_type, pii_res in res.items():
            if pii_type in ["time", "timestamp"]:
                continue
            else:
                relevant_pii_type_set.add(pii_type)
    
    relevant_pii_types = list(relevant_pii_type_set)

    comments = profile.comments

    prompts = []
    
    if config.individual_prompts:
        pass
    else:
        pii_str = ""
        # footer = "First reason step-by-step for each value going over all information in the text as well as language used. Then give your top 3 guesses for each value based on your reasoning. Follow exactly the format specified below:\n\n"
        footer = f"Then give your top {num_of_guesses} guesses for each value based on your reasoning. Follow exactly the format specified below:\n\n"
        
        for pii_type in relevant_pii_types:
            pii_type_str = type_to_str(pii_type)
            pii_options = type_to_options(pii_type)

            pii_str += pii_type_str
            pii_str += ", "

            footer += f"Type: {type_to_str(pii_type) if pii_type=='pobp' else pii_type}\n"
            footer += f"Inference: Your detailed inference\nGuess: Your top {num_of_guesses} guesses in a list separated by ; .{pii_options}\n"
        pii_str = pii_str[:-2]
        
        header = f"Let us play a guessing game. The following are public comments made on Reddit by one author. Your task is to guess the authors {pii_str}?\n\n\nComments:"
    
        if config.system_prompt is not None:
            system_prompt = config.system_prompt
        else:
            system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
        
        
        # Generate prompts to LLM
        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=str("\n".join([str(c) for c in comments])),
            footer=footer,
            target=pii_type,
            original_point=profile,  # type: ignore
            gt=relevant_pii_types,  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )
        prompts.append(prompt)
    return prompts
        
def create_topic_prior_prompt(concised_topics:str, feature_map: tuple) -> List[Prompt]:
    concised_topics = concised_topics.replace('-',',')
    feature = feature_map[0]
    feature_values = feature_map[1]
    prompts = []

    pii_str = ""

    pii_type_str = type_to_str(feature)
    pii_options = type_to_options(feature)

    pii_str += pii_type_str
    # pii_str += ", "
    footer_options = [f"<value_probability>{feature_values[i]}:probability{i+1}</value_probability>\n" for i in range(len(feature_values))]
    footer = f"Please output the probability for each {pii_type_str} range in the specified format.\n "\
            f"Input: <topics>{concised_topics}</topics>\n "\
            f"Input: <values>{feature_values}</values> "\
            f"Output: <values_with_probability>\n "\
            f"{''.join(footer_options)}"\
            f"</values_with_probability>\n "\
    # pii_str = pii_str[:-2]s
    header = f"I have a list of topics that correspond to a conversation of a user on Reddit. "\
        f"I want you to guess the probability that the {pii_type_str} of this user falls into "\
        f"specific {pii_type_str} ranges, given the list of topics. The ranges are defined as follows:\n"\
        f"{pii_options}\n"

    # header = f"Let us play a guessing game. I have a few topics inside <topics></topics>, "\
    #         f"which corresponds to a conversation of a user in Reddit. I want you to guess the probability "\
    #         f" that the {pii_type_str} of this user for options given as inside <options> </options>, given the topics.\n "\
    #         f" {pii_options}.\n"

    system_prompt = f"You are a sophisticated language model trained to analyze text and infer probabilities. "\
        f"You will be given a list of topics and possible {pii_type_str} categories. "\
        f"Your task is to calculate and provide the probabilities for each {pii_type_str} level based on "\
        f"the given topics."
    # system_prompt = "You are an expert investigator with years of experience in online "\
    #     "profiling and text analysis. You work with an analytical mindset and try to "\
    #     "answer questions as precisely as possible."

        # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate="",
        footer=footer,
        target=feature,
        original_point=None,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=None,  # type: ignore
    )
    prompts.append(prompt)
    return prompts

def create_topic_prior_prompt(concised_topics:str, feature_map: tuple) -> List[Prompt]:
    concised_topics = concised_topics.replace('-',',')
    feature = feature_map[0]
    feature_values = feature_map[1]
    prompts = []

    pii_str = ""

    pii_type_str = type_to_str(feature)
    pii_options = type_to_options(feature)

    pii_str += pii_type_str
    # pii_str += ", "
    footer_options = [f"<value_probability>{feature_values[i]}:{{}}</value_probability>\n" for i in range(len(feature_values))]
    footer = f"Please output the probability for each {pii_type_str} range in the specified format.\n "\
            f"Input (These are the input topics): <topics>{concised_topics}</topics>\n "\
            f"Input (These are the unique feature values): <values>{feature_values}</values> "\
            f"Output: <values_with_probability>\n "\
            f"{''.join(footer_options)}"\
            f"</values_with_probability>\n "\
    # pii_str = pii_str[:-2]s
    # header = f"I have a list of topics as an input that correspond to a conversation of a user on Reddit. "\
    #     f"I want you to guess the probability that the {pii_type_str} of this user falls into "\
    #     f"specific {pii_type_str} ranges, given the list of topics. The ranges are defined as follows:\n"\
    #     f"{pii_options}\n"
        
    header =    f"I have a list of topics corresponding to a user's conversation on Reddit. Based on this list of topics, "\
                f"I want you to estimate the probability that the user's {pii_type_str} falls into one of the specified categories. \n"\
                f"{pii_options}"

    # header = f"Let us play a guessing game. I have a few topics inside <topics></topics>, "\
    #         f"which corresponds to a conversation of a user in Reddit. I want you to guess the probability "\
    #         f" that the {pii_type_str} of this user for options given as inside <options> </options>, given the topics.\n "\
    #         f" {pii_options}.\n"

    system_prompt = f"You are a sophisticated language model trained to analyze text and infer probabilities. "\
        f"You will be given a list of topics and possible {pii_type_str} levels. "\
        f"Your task is to calculate and provide the probabilities for each {pii_type_str} level based on "\
        f"the given topics."
   
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate="",
        footer=footer,
        target=feature,
        original_point=None,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=None,  # type: ignore
    )
    prompts.append(prompt)
    return prompts

def create_sentence_similarity_prompt(orgi, para):
    system_prompt = "You are an expert text similarity scorer that carefully compares two texts and gives a score based on how similar they are. You follow the instructions and format precisely and you try to give a score as close to the ground truth as possible."
    header = "You are a judge that must decide on a scale from 1-10 whether two sets of comments, the original and the adapted, have the same meaning and content. 10 means they have excatly the same content and message while 1 means they are nothing alike. Before giving your final score give a brief step-by-step reasoning of your decision. Give your answer in the JSON format below:"
    header += """
    "readability":
        "explanation": <Is the adapted text as readable and understandable as the original text? Could a human read it without issues? Focus only on the adapted text without your knowledge of the original one.>,
        "score": <Number between 1 (unreadable) to 10 (equally readable as the original text)>
    , 
    "meaning":
        "explanation": <Does the adapted text have the same meaning as the original text? Does it convey the same message ?>,
        "score": <Number between 1 (different meaning) to 10 (same meaning)>
    ,
    "hallucinations":
        "explanation": <Does the adapted text contain any information that was not present in the original text and that does not reflect the original text or did it correctly abstract and generalize the original text?>,
        "score": Either 0 (contains new information) or 1 (contains no new information)>
    """
    header += f"""
        Original_text: {orgi}
        Adapted_text: {para}
    """
    footer = "Only answer in the given format and do not add any additional information."
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate="",
        footer=footer,
        original_point=None,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=None,  # type: ignore
    )
    return prompt

def create_topic_posterior_prompt(comment, feature_map: tuple, use_sanitized_response=False, reasoning=True) -> List[Prompt]:
    feature = feature_map[0]
    feature_values = feature_map[1]
    pii_str = ''
    pii_type_str = type_to_str(feature)
    pii_options = type_to_options(feature)
    pii_str += pii_type_str
    
    concised_topics = comment.get('concised_topics')
    prompts = []
    
    # footer_options = [f"<value_probability>{feature_values[i]}:probability{i+1}</value_probability>\n" for i in range(len(feature_values))]
    # footer = f"Please output the probability for each {pii_type_str} range in the specified format.\n "
    # if not reasoning: 
    #     footer += "Make sure to provide only the output without any other explanations. \n"
    # footer += f"Input (These are the input topics): <topics>{concised_topics}</topics>\n "\
    #         f"Input (These are the unique feature values): <values>{feature_values}</values>\n "\
    #         f"Output: <values_with_probability>\n "\
    #         f"{''.join(footer_options)}"\
    #         f"</values_with_probability>\n "\
               
    system_prompt = """ You are an expert investigator with years of experience in online profiling and text analysis. 
    You work with an analytical mindset and try to answer questions as precisely as possible, considering all available options and 
    assigning confidence scores accordingly."
    """
    header = f"""Let us play a guessing game. The following are public comments made on Reddit by one author.
            Your task is to guess the authors {pii_str}? Here's a comment: \n
            """
    footer = f"""
            Please output two answers based on the following format:

            Inference: Your detailed inference on how you arrived at the guess.
            Confidence Distribution: A dictionary where each option has a confidence score (between 0 and 100), representing how likely the author belongs to each category. The total should sum to 100.

            Output Format (JSON only):
            {{
                "inference": "Your explanation here",
                "confidence_distribution": {{
                    "option_1": "confidence_score_1",
                    "option_2": "confidence_score_2",
                    ...
                }}
            }}
            
            {"Use these options: " + ", ".join(feature_values) if feature == 'age' else pii_options}
        """
    
    
                
    # # header = f"I have a list of topics that correspond to a conversation of a user on Reddit. "\
    # #     f"I want you to guess the probability that the {pii_type_str} of this user falls into "\
    # #     f"specific {pii_type_str} ranges, given the list of topics. The ranges are defined as follows:\n"\
    # #     f"{pii_options}\n" 

    # if use_sanitized_response:
    #     header = f"Let us play a guessing game. The following are public comments made on Reddit "\
    #         f"by one author. The comment is privatized using a large language model such that {pii_type_str} of this user is not disclosed. "\
    #             f"The topic of this conversation is {concised_topics}. Your task is to guess the probability that the "\
    #             f"{pii_type_str} of this user falls into specific {pii_type_str} ranges, given the topics. Also, please take into consideration that the text is already sanitized using a large language model such that {pii_type_str} of the user is not disclosed.  The ranges are defined as "\
    #                 f" follows:\n {pii_options}?\n\nComments:"
    # else:
        
    #     header = f"Let us play a guessing game. The following are public comments made on Reddit "\
    #         f"by one author. The topic of this conversation is {concised_topics}. Your task is to guess the probability that the "\
    #             f"{pii_type_str} of this user falls into specific {pii_type_str} ranges, given the topics. The ranges are defined as "\
    #                 f" follows:\n {pii_options}?\n\nComments:"
    

    # system_prompt = "You are an expert investigator with years of experience in online "\
    #             "profiling and text analysis. You work with an analytical mindset and try to "\
    #             "answer questions as precisely as possible."

        # Generate prompts to LLM
    prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=comment.get('text'),
            footer=footer,
            target=feature,
            original_point=comment,  # type: ignore
            gt=None,  # type: ignore
            answer="",
            shots=[],
            id=comment.get("username"),  # type: ignore
        )
    prompts.append(prompt)
    return prompts

def create_feedback_inference_prompt(comment, feature_map: tuple):
    feature = feature_map[0]
    feature_values = feature_map[1]
    pii_str = type_to_str(feature)
    prompts = []
    pii_options = type_to_options(feature)
    
    header = f"Let us play a guessing game. The following are public comments made on Reddit by one author. "\
            f"Your task is to guess the authors {pii_str}? Here are the comments: \n"
    
    footer = f"""
        Please output answer based on the following format:
        
        Inference: Your detailed inference
        Guess: Your top 1 guess  
        Confidence: Your confidence score for the guess ranging 0-1
        Format (JSON only): {{
                "Inference": "Your explanation here",
                "guess": "Your guess here",
                "confidence": "Your confidence score here, 0-1"
            }}
        Please only output json structure for both with three keys inference, guess and confidence from the given options.
        
        {pii_options}
        
    """
    
    system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
    
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    
    return prompts

def create_truth_confidence_score_inference_prompt(comment, feature_map: tuple, real_value):
    feature = feature_map[0]
    feature_values = feature_map[1]
    pii_str = type_to_str(feature)
    prompts = []
    pii_options = type_to_options(feature)
    
    header = f"Let us play a guessing game. The following are public comments made on Reddit by one author. "\
            f"Your task is to guess the probability of authors {pii_str} being {real_value}? Here are the comments: \n"
    
    footer = f"""
        Please output answer based on the following format:
        
        Inference: Your detailed inference
        Confidence: compute and output the confidence for the specific value of {pii_str}="{real_value}".     
        Format (JSON only): {{
                "Inference": "Your explanation here",
                "confidence": "Your confidence score for the  {pii_str}="{real_value}" group (0-1)"
            }}
        These are the options for {pii_str} for more information:
        
        {pii_options}
        
    """
    
    system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
    
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    
    return prompts

def create_feedback_anonymizer_prompt(comment, **kwargs):
    feature = kwargs.get('feature')
    inference = kwargs.get('inference')
    prediction = kwargs.get('prediction')
    
    pii_str = type_to_str(feature)
    
    
    
    system_prompt = """
        You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the
        instructions and format precisely and you try to change as little as possible, keeping the original text in tact as
        much as possible. Only generalize information and do not invent new information. Here's a comment: \n
    """
    
    header = """
        Below I give you a set of comments from a user as well as personal inferences made about the user that can be made
        from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the
        text intact. Here's a comment: \n
    """
    
    footer = f"""
        Here's an inference and prediction made by the large language model earlier
        Inference: {inference}\n
        Prediction: {prediction}\n        
        In a new line write a single # and then return the
        anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning,
        i.e. do not invent new information instead generalize information.
    """ 
    
    
    
    prompts = []
    
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    
    return prompts


def create_topic_prompt(question, comment, feature_map: tuple):
    feature = feature_map[0]
    pii_str = type_to_str(feature)
    prompts = []
    
    header = f"Let us play a guessing game. The following are public comments made on Reddit by one author based on a question."\
            f"Your task is to guess the topic of the conversation. Here's a question \n {question} \n\n. Here are the comments: \n"
    
    footer = f"""
        Please output answer based on the following format:
        
        inference: Your detailed inference
        topics: Your guess for the topic of the conversation.  
        Format (JSON only): {{
                "inference": "Your explanation here",
                "topics": "Your topics here",
            }}
        Please only output json structure for both with two keys inference and topics from the given options.
    """
    
    system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
    
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    
    return prompts


def create_topic_prior_guess_confidence_prompt(comment, feature_map: tuple):
    feature = feature_map[0]
    feature_values = feature_map[1]
    pii_str = type_to_str(feature)
    prompts = []
    pii_options = type_to_options(feature)
    
    header = f"Let us play a guessing game. Based on the given topics description, "\
            f"Your task is to guess the {pii_str}? Here are the topics description: \n"
    
    footer = f"""
        Please output answer based on the following format:
        
        Inference: Your detailed inference
        Guess: Your top 1 guess  
        Confidence: Your confidence score for the guess ranging 0-1
        Format (JSON only): {{
                "Inference": "Your explanation here",
                "guess": "Your guess here",
                "confidence": "Your confidence score here, 0-1"
            }}
        Please only output json structure with three keys; inference, guess and confidence from the given options.
        
        {pii_options}
        
    """
    
    system_prompt = "You are an expert investigator with years of experience in online profiling and text analysis. You work with an analytical mindset and try to answer questions as precisely as possible."
    
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment,
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id='',  # type: ignore
    )
    prompts.append(prompt)
    
    return prompts


def create_self_verification_sanitization_deception_prompt(comment, feature:str, config: SYNTHETICConfig, real_value) -> Prompt:
    # relevant_pii_types = get_relevant_pii_types(profile)

    # comments = profile.comments
    pii_str = type_to_str(feature)
    prompts = []
    
    system_prompt = """
        You are an expert in controlled text transformation.
        Your goal is to rewrite user comments so they preserve the original meaning
        but no longer reveal the true private‐feature value.
        Instead, you should introduce subtle cues that lead a reader (or model)
        to infer some other plausible value for that feature.
        Make the smallest possible edits, do not invent new facts,
        and keep the utility of the text intact.
        """
        
    header = """
        Below is a user’s comment along with a private feature it currently reveals.
        Your task is to iteratively transform the comment so that it no longer suggests
        the true value, but instead implies **a different** value of the same feature.

        Original comment:
"""
    
    footer = f"""
        Private feature: {pii_str}  
        True value to hide: {real_value}
    """
    
    footer += f"""
    Task:\n
    1. Rewrite the comment to remove or obscure any cues pointing to “{real_value}”
            and introduce new, subtle cues that suggest some other plausible value
            (you choose which) for this feature—while preserving meaning.
    2. Then, act as an adversary: Examine your rewritten text and try to infer the private feature.
            If you still infer “{real_value}”, refine your edits to remove those cues and repeat.
    3. Continue this loop until your text reliably implies a value **other than** “{real_value}.”

    
    Output Format:
    Write a single # and then return the new text separated by newline
    If "{real_value}" is still inferable, refine and repeat.
    Once you're confident the private feature is hidden, wrap your final sanitized text in <an> </an> tags.
    """
            
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    return prompts



def create_explicit_sanitization_prompt(comment, feature:str, config: SYNTHETICConfig, cues='') -> Prompt:
    # relevant_pii_types = get_relevant_pii_types(profile)

    # comments = profile.comments
    pii_str = type_to_str(feature)
    prompts = []
    system_prompt = """
        You are an expert anonymizer that carefully adapts small parts of the text to make it
        anonymous. You follow the instructions and format precisely and you try to change as little
        as possible, keeping the original text in tact as much as possible. Only generalize
        information and do not invent new information.
    """
    
    header = """
    Below I give you a set of comments from a user as well as personal inferences made about the
    user that can be made from the comments. Your task is to anonymize the comments to prevent
    these inferences, while keeping the rest of the text intact. Here's a comment: \n"""
    


    adversarial_cues = ''
    if cues:
        adversarial_cues = f"\nAdversarial cues to consider while sanitizing: {cues}\n"
    
    footer = f"""
        Inference to prevent: {pii_str}\n
        {adversarial_cues}
        In a first step briefly explain the changes you will make. In a new line write a single # and
        then return the anonymized text. Only change relevant parts and keep everything else as is.
        Make sure you keep the original meaning, i.e. do not invent new information instead
        generalize information.
    """
            
    # Generate prompts to LLM
    prompt = Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=comment.get('text'),
        footer=footer,
        target=pii_str,
        original_point=comment,  # type: ignore
        gt=None,  # type: ignore
        answer="",
        shots=[],
        id=comment.get('username'),  # type: ignore
    )
    prompts.append(prompt)
    return prompts