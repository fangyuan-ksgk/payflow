from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.aor import AOR, load_aors
from src.utils import get_oai_response, parse_json_response
from dataclasses import dataclass, field

FUNCTION_CALL_PROMPT = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for AORs based on a specific item or keyword.
2. search_aor_with_no: Searches for an AOR using its unique identification number.
3. query_detail: Performs a detailed query on a specific AOR, accessing its full text content.
4. query_rough: Executes a quick query on an AOR using a concise summary of its information.

User query: {user_query}
"""


def search_aor_with_no(query_item: str, aor_list: List[AOR]):
    """ 
    Slot in an AOR number and return the relevant AOR
    """
    matching_aors = aor_list
    prefix_length = 4
    
    while len(matching_aors) > 1 and prefix_length <= len(query_item):
        matching_aors = [aor for aor in matching_aors if query_item[:prefix_length].lower() == aor.no[:prefix_length].lower()]
        prefix_length += 2
    
    if not matching_aors:
        # If no exact match found, fall back to the original method
        matching_aors = [aor for aor in aor_list if query_item.lower() in aor.no.lower()]
    
    return matching_aors[:3]  # Return up to 3 matching AORs

def search_aor_with_item(query_item: str, aor_list: List[AOR], top_k: int = 1, threshold: float = 0.4):
    """ 
    Slot in a Key-word based query (regarding the item to be claimed) and return a relevant AOR
    """

    # Implement a faster keyword matching algorithm
    def keyword_match(query_item: str, items: List[str]) -> bool:
        query_words = set(query_item.lower().split())
        for item in items:
            item_words = set(item.lower().split())
            if query_words.intersection(item_words):
                return True
        return False

    # Filter AORs based on keyword matching
    matching_aors = [aor for aor in aor_list if keyword_match(query_item, aor.items)]

    # If no matches found, fall back to the original method
    if not matching_aors:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the query_item and AOR descriptions
        query_embedding = model.encode([query_item])

        aor_description_embeddings = model.encode([aor.description for aor in aor_list]) # Description-base search
        similarities = cosine_similarity(query_embedding, aor_description_embeddings)[0]
        top_indices = [i for i in np.argsort(similarities)[::-1] if similarities[i] >= threshold][:top_k]

        # Item-based search
        aor_item_embeddings = []
        item_indices = []
        for i, aor in enumerate(aor_list):
            for item in aor.items:
                item_embedding = model.encode([item])
                aor_item_embeddings.append(item_embedding)
                item_indices.append(i)
        
        item_similarities = cosine_similarity(query_embedding, np.vstack(aor_item_embeddings)).flatten()
        
        item_top_indices = [item_indices[i] for i, sim in enumerate(item_similarities) if sim >= threshold]
        
        # Combine results from description-based and item-based search
        top_indices = list(set(top_indices + item_top_indices))

        matching_aors = [aor_list[i] for i in top_indices]

    return matching_aors


def query_rough(aor, query):
    """ 
    Query rough (structured) information of AOR 
    """
    text = aor.narrative
    
    QUERY_TEMPLATE = """
    Given the following AOR:
    {txt}
    Answer the following question:
    {query}
    """
    query_prompt = QUERY_TEMPLATE.format(txt=text, query=query)
    
    response = get_oai_response(query_prompt)
    
    return response


def query_detail(aor, query):
    """ 
    Naive implementation | TBD: structured RAG with cached dictionary
    ICL > RAG given enough context @DeepMind Research
    """
    text = aor.pdf_text 
    
    QUERY_TEMPLATE = """
    Given the following AOR:
    {txt}
    Answer the following question:
    {query}
    """
    query_prompt = QUERY_TEMPLATE.format(txt=text, query=query)
    
    response = get_oai_response(query_prompt)
    
    return response


@dataclass 
class Memory:
    
    aor_list: List[AOR] = field(default_factory=list)
    all_aors: List[AOR] = field(default_factory=load_aors)
    
    def reset(self, aor_list):
        self.aor_list = aor_list
        
    def search_aor_with_item(self, query_item: str):
        aor_list = search_aor_with_item(query_item, self.all_aors)
        self.reset(aor_list)
        return f"Found AOR{aor_list[0].no} related to the item {query_item}"
    
    def search_aor_with_no(self, query_no: str):
        aor_list = search_aor_with_no(query_no, self.all_aors)
        self.reset(aor_list)
        return f"Found AOR{aor_list[0].no} related to the query number {query_no}"
        
    def query_detail(self, query):
        aor = self.aor_list[0]
        return query_detail(aor, query)
    
    def query_rough(self, query):
        aor = self.aor_list[0]
        return query_rough(aor, query)
    
    
FUNCTION_CALL_PROMPT = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for AORs based on a specific item or keyword.
2. search_aor_with_no: Searches for an AOR using its unique identification number.
3. query_detail: Performs a detailed query on a specific AOR, accessing its full text content.
4. query_rough: Executes a quick query on an AOR using a concise summary of its information.

User query: {user_query}
"""


def query_memory(user_query, memory: Memory):
    """ 
    Query response will be conduced together with external Memory state
    """

    # Call Response 
    # Missing a logic here to determine whether to limit option to searching within retrieved AORs 
    user_query = "Find AORs related to hardware"
    call_prompt = FUNCTION_CALL_PROMPT.format(user_query=user_query)
    response = get_oai_response(call_prompt)
    response_dict = parse_json_response(response)
        
    # Extract function name and query
    function_name = response_dict['function_name']
    query = response_dict['query']
    
    all_aors = load_aors()
    # Call the appropriate function based on the response
    if function_name == "search_aor_with_item":
        search_str = memory.search_aor_with_item(query)
        return search_str, memory
    elif function_name == "search_aor_with_no":
        search_str = memory.search_aor_with_no(query)
        return search_str, memory
    elif function_name == "query_detail":
        return memory.query_detail(query), memory
    elif function_name == "query_rough":
        return memory.query_rough(query), memory
    else:
        return "Invalid function name"