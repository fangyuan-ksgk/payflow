from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.aor import AOR 
from src.utils import get_oai_response



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