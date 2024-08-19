from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from src.aor import AOR, load_aors, Invoice, load_invoices
from src.prompt import SYSTEM_PROMPT
from src.utils import get_oai_response, parse_json_response
from dataclasses import dataclass, field


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

def search_invoice_with_no(query_item: str, invoice_list: List[str]):
    """ 
    Slot in an invoice number and return the relevant invoice
    """
    matching_invoices = invoice_list
    prefix_length = 4
    
    while len(matching_invoices) > 1 and prefix_length <= len(query_item):
        matching_invoices = [invoice for invoice in matching_invoices if query_item[:prefix_length].lower() == invoice.no[:prefix_length].lower()]
        prefix_length += 2
    
    if not matching_invoices:
        # If no exact match found, fall back to the original method
        matching_invoices = [invoice for invoice in invoice_list if query_item.lower() in invoice.no.lower()]
    
    return matching_invoices[:3]  # Return up to 3 matching invoices

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


def search_invoice_with_item(query_item: str, invoice_list: List[str], top_k: int = 1, threshold: float = 0.4):
    """ 
    Slot in a Key-word based query (regarding the item to be claimed) and return a relevant invoice
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
    matching_invoices = [invoice for invoice in invoice_list if keyword_match(query_item, invoice.items)]

    # If no matches found, fall back to the original method
    if not matching_invoices:
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the query_item and Invoice descriptions
        query_embedding = model.encode([query_item])

        invoice_description_embeddings = model.encode([invoice.description for invoice in invoice_list]) # Description-base search
        similarities = cosine_similarity(query_embedding, invoice_description_embeddings)[0]
        top_indices = [i for i in np.argsort(similarities)[::-1] if similarities[i] >= threshold][:top_k]

        # Item-based search
        invoice_item_embeddings = []
        item_indices = []
        for i, invoice in enumerate(invoice_list):
            for item in invoice.items:
                item_embedding = model.encode([item])
                invoice_item_embeddings.append(item_embedding)
                item_indices.append(i)
        
        item_similarities = cosine_similarity(query_embedding, np.vstack(invoice_item_embeddings)).flatten()
        
        item_top_indices = [item_indices[i] for i, sim in enumerate(item_similarities) if sim >= threshold]
        
        # Combine results from description-based and item-based search
        top_indices = list(set(top_indices + item_top_indices))

        matching_invoices = [invoice_list[i] for i in top_indices]

    return matching_invoices


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
    
    response = get_oai_response(query_prompt, system_prompt=SYSTEM_PROMPT)
    
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
    Keep your answer concise and to the point.
    """
    query_prompt = QUERY_TEMPLATE.format(txt=text, query=query)
    
    response = get_oai_response(query_prompt, system_prompt=SYSTEM_PROMPT)
    
    return response

def query_invoice_detail(invoice, query):
    """ 
    Naive implementation | TBD: structured RAG with cached dictionary
    ICL > RAG given enough context @DeepMind Research
    """
    text = invoice.invoice_text
    QUERY_TEMPLATE = """ 
    Given the following invoice:
    {txt}
    Answer the following question:
    {query}
    Keep your answer concise and to the point.
    """ 
    query_prompt = QUERY_TEMPLATE.format(txt=text, query=query)
    
    response = get_oai_response(query_prompt, system_prompt=SYSTEM_PROMPT)
    
    return response

@dataclass 
class Memory:
    
    aor_list: List[AOR] = field(default_factory=list)
    all_aors: List[AOR] = field(default_factory=load_aors)
    invoice_list: List[Invoice] = field(default_factory=list)
    all_invoices: List[Invoice] = field(default_factory=load_invoices)
    messages: List[str] = field(default_factory=list)
    last_call = []

    @property
    def narrative(self):
        if len(self.aor_list)==0:
            return ""
        return self.aor_list[0].narrative
    
    @property
    def invoice_narrative(self):
        if len(self.invoice_list)==0:
            return ""
        return self.invoice_list[0].narrative
        
    @property
    def invoice_image(self):
        if len(self.invoice_list)==0:
            return ""
        return self.invoice_list[0].image
        
    @property
    def aor_image(self):
        if len(self.aor_list)==0:
            return ""
        return self.aor_list[0].image
    
    def reset(self, aor_list = [], invoice_list = []):
        self.aor_list = aor_list
        self.invoice_list = invoice_list
        
    def search_aor_with_item(self, query_item: str):
        aor_list = search_aor_with_item(query_item, self.all_aors)
        self.reset(aor_list = aor_list, invoice_list = self.invoice_list)
        self.last_call.append("search_aor")
        if aor_list:
            return f"Found AOR{aor_list[0].no} related to the item {query_item}"
        else:
            return f"No AOR found related to the item {query_item}"
    
    def search_aor_with_no(self, query_no: str):
        aor_list = search_aor_with_no(query_no, self.all_aors)
        self.reset(aor_list = aor_list, invoice_list = self.invoice_list)
        self.last_call.append("search_aor")
        if aor_list:
            return f"Found AOR{aor_list[0].no} related to the query number {query_no}"
        else:
            return f"No AOR found related to the query number {query_no}"
    
    def search_invoice_with_item(self, query_item: str):
        invoice_list = search_invoice_with_item(query_item, self.all_invoices)
        self.reset(aor_list = self.aor_list, invoice_list = invoice_list)
        self.last_call.append("search_invoice")
        if invoice_list:
            return f"Found Invoice{invoice_list[0].no} related to the item {query_item}"
        else:
            return f"No Invoice found related to the item {query_item}"
    
    def search_invoice_with_no(self, query_no: str):
        invoice_list = search_invoice_with_no(query_no, self.all_invoices)
        self.reset(aor_list = self.aor_list, invoice_list = invoice_list)
        self.last_call.append("search_invoice")
        if invoice_list:
            return f"Found Invoice{invoice_list[0].no} related to the query number {query_no}"
        else:
            return f"No Invoice found related to the query number {query_no}"
        
    def query_detail(self, query):
        aor = self.aor_list[0]
        return query_detail(aor, query)
    
    def query_invoice_detail(self, query):
        invoice = self.invoice_list[0]
        return query_invoice_detail(invoice, query)
    
    def update_user_response(self, response, temp = False):
        if temp:
            return self.messages + [{"role": "user", "content": response}]
        else:
            self.messages.append({"role": "user", "content": response})
            return self.messages
    
    def update_agent_response(self, response, temp = False):
        if temp:
            return self.messages + [{"role": "assistant", "content": response}]
        else:
            self.messages.append({"role": "assistant", "content": response})
    
    
INITIAL_SEARCH_PROMPT = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for AORs based on a specific item or keyword.
2. search_aor_with_no: Searches for an AOR using its unique identification number.
3. search_invoice_with_item: Searches for invoices based on a specific item or keyword.
4. search_invoice_with_no: Searches for an invoice using its unique identification number.

User query: {user_query}
"""

CONTINUE_SEARCH_PROMPT = """
Based on the user's query and the retrieved AOR and Invoice which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for another AOR based on a specific item or keyword. Only use this if current AOR is not relevant.
2. search_aor_with_no: Searches for another AOR using its unique identification number. Only use this if current AOR is not relevant.
3. search_invoice_with_item: Searches for another invoice based on a specific item or keyword. Only use this if current invoice is not relevant.
4. search_invoice_with_no: Searches for another invoice using its unique identification number. Only use this if current invoice is not relevant.
5. query_detail: Performs a detailed query on current AOR, accessing its full text content.
6. query_invoice_detail: Performs a detailed query on current invoice, accessing its full text content.
7. direct_answer: Provide your answer to the query base on current information.

Retrieved Invoice: {invoice_narrative}

Retrieved AOR: {aor_narrative}

User query: {user_query}
"""

CONTINUE_INVOICE_SEARCH_PROMPT = """
Based on the user's query and the retrieved AOR and Invoice which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_invoice_with_item: Searches for another invoice based on a specific item or keyword. Only use this if current invoice is not relevant.
2. search_invoice_with_no: Searches for another invoice using its unique identification number. Only use this if current invoice is not relevant.
3. query_detail: Performs a detailed query on current AOR, accessing its full text content.
4. query_invoice_detail: Performs a detailed query on current invoice, accessing its full text content.
5. direct_answer: Provide your answer to the query base on current information. Use this when other functions are not relevant.

Retrieved AOR: {aor_narrative}

Retrieved Invoice: {invoice_narrative}

User query: {user_query}
"""

CONTINUE_INVOICE_SEARCH_PROMPT_NO_AOR = """
Based on the user's query and the retrieved Invoice which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_invoice_with_item: Searches for another invoice based on a specific item or keyword. Only use this if current invoice is not relevant.
2. search_invoice_with_no: Searches for another invoice using its unique identification number. Only use this if current invoice is not relevant.
3. query_invoice_detail: Performs a detailed query on current invoice, accessing its full text content.
4. direct_answer: Provide your answer to the query base on current information.

Retrieved Invoice: {invoice_narrative}

User query: {user_query}
"""

CONTINUE_INVOICE_SEARCH_PROMPT_NO_AOR_NO_INVOICE = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_invoice_with_item: Searches for an invoice based on a specific item or keyword.
2. search_invoice_with_no: Searches for an invoice using its unique identification number.
3. direct_answer: Provide your answer to the query base on current information.

User query: {user_query}
"""

CONTINUE_INVOICE_SEARCH_PROMPT_NO_INVOICE = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_invoice_with_item: Searches for an invoice based on a specific item or keyword.
2. search_invoice_with_no: Searches for an invoice using its unique identification number.
3. direct_answer: Provide your answer to the query base on current information.

Retrieved AOR: {aor_narrative}

User query: {user_query}
"""

CONTINUE_AOR_SEARCH_PROMPT = """
Based on the user's query and the retrieved AOR and Invoice which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for another AOR based on a specific item or keyword. Only use this if current AOR is not relevant.
2. search_aor_with_no: Searches for another AOR using its unique identification number. Only use this if current AOR is not relevant.
3. query_detail: Performs a detailed query on current AOR, accessing its full text content.
4. query_invoice_detail: Performs a detailed query on current invoice, accessing its full text content.
5. direct_answer: Provide your answer to the query base on current information.

Retrieved AOR: {aor_narrative}

Retrieved Invoice: {invoice_narrative}

User query: {user_query}
"""

CONTINUE_AOR_SEARCH_PROMPT_NO_INVOICE = """
Based on the user's query and the retrieved AOR which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for another AOR based on a specific item or keyword. Only use this if current AOR is not relevant.
2. search_aor_with_no: Searches for another AOR using its unique identification number. Only use this if current AOR is not relevant.
3. query_detail: Performs a detailed query on current AOR, accessing its full text content.
4. direct_answer: Provide your answer to the query base on current information.

Retrieved AOR: {aor_narrative}

User query: {user_query}
"""


CONTINUE_AOR_SEARCH_PROMPT_NO_AOR = """
Based on the user's query and the retrieved Invoice which may or may not be relevant, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for another AOR based on a specific item or keyword.
2. search_aor_with_no: Searches for another AOR using its unique identification number.
3. query_invoice_detail: Performs a detailed query on current invoice, accessing its full text content.
4. direct_answer: Provide your answer to the query base on current information.

Retrieved Invoice: {invoice_narrative}

User query: {user_query}
"""

CONTINUE_AOR_SEARCH_PROMPT_NO_AOR_NO_INVOICE = """
Based on the user's query, determine which function should be called and provide the appropriate query.
Output your response as a JSON object with the following structure:
{{
    "function_name": "<name of the function to call>",
    "query": "<query to pass to the function>"
}}

Available functions:
1. search_aor_with_item: Searches for another AOR based on a specific item or keyword.
2. search_aor_with_no: Searches for another AOR using its unique identification number.
3. direct_answer: Provide your answer to the query base on current information.

User query: {user_query}
"""

DIRECT_ANSWER_PROMPT = """
Provide your answer to the query base on current information.

Output your response as a JSON object with the following structure:
{{
    "function_name": "direct_answer",
    "query": "your response to the user query"
}}

User query: {user_query}
"""

DIRECT_ANSWER_TEMPLATE = """ 
Base on what you've found, provide your answer to the user query. Do you make up information, mention you did not find relevant information if the provided information is not relevant.

Retrieved Invoice: {invoice_narrative}

Retrieved AOR: {aor_narrative}

User query: {user_query}

Provide your thought and answer. For instance: 
Thought: xxx
Answer: xxx
"""


def route_query(user_query, memory: Memory, first_query: bool) -> str:
    """ 
    Route the query to the appropriate prompt based on the last function called by the agent
    """
    first_query = not memory.last_call
    aor_searched = "search_aor" in memory.last_call
    invoice_searched = "search_invoice" in memory.last_call
    has_aor = memory.narrative != ""
    has_invoice = memory.invoice_narrative != ""
    
    if first_query:
        return INITIAL_SEARCH_PROMPT.format(
            aor_narrative=memory.narrative, invoice_narrative=memory.invoice_narrative, user_query=user_query
        )
    else:
        if aor_searched and not invoice_searched:
            if has_aor and has_invoice:
                return CONTINUE_INVOICE_SEARCH_PROMPT.format(
                    aor_narrative=memory.narrative, invoice_narrative=memory.invoice_narrative, user_query=user_query
                )
            elif has_aor and not has_invoice:
                return CONTINUE_INVOICE_SEARCH_PROMPT_NO_INVOICE.format(
                    aor_narrative=memory.narrative, user_query=user_query
                )
            elif not has_aor and has_invoice:
                return CONTINUE_INVOICE_SEARCH_PROMPT_NO_AOR.format(
                    invoice_narrative=memory.invoice_narrative, user_query=user_query
                )
            else:
                return CONTINUE_INVOICE_SEARCH_PROMPT_NO_AOR_NO_INVOICE.format(
                    user_query=user_query
                )  
        elif invoice_searched and not aor_searched:
            if has_invoice and has_aor:
                return CONTINUE_AOR_SEARCH_PROMPT.format(
                    aor_narrative=memory.narrative, invoice_narrative=memory.invoice_narrative, user_query=user_query
                )
            elif has_invoice and not has_aor:
                return CONTINUE_AOR_SEARCH_PROMPT_NO_AOR.format(
                    invoice_narrative=memory.invoice_narrative, user_query=user_query
                )
            elif not has_invoice and has_aor:
                return CONTINUE_AOR_SEARCH_PROMPT_NO_INVOICE.format(
                    aor_narrative=memory.narrative, user_query=user_query
                )
            else:
                return CONTINUE_AOR_SEARCH_PROMPT_NO_AOR_NO_INVOICE.format(
                    user_query=user_query
                )
        else:
            return DIRECT_ANSWER_PROMPT.format(
                invoice_narrative=memory.invoice_narrative, aor_narrative=memory.narrative, user_query=user_query
            )
            
            
def parse_thought_answer(response_str):
    """
    Parse the thought and answer from the response string
    """
    thought_str = response_str.split("Thought:")[1].split("Answer:")[0].strip()
    answer_str = response_str.split("Answer:")[1].strip()
    if not answer_str:
        return "", response_str
    return thought_str, answer_str


def query_memory_single(user_query, memory: Memory) -> tuple[str, Memory, bool]:
    """ 
    Query response will be conduced together with external Memory state
    Return : 
    -- info_str: String
    -- memory: Memory
    -- bool: Terminate
    """

    # Call Response 
    call_prompt = route_query(user_query, memory, first_query=True)

    response = get_oai_response(memory.update_user_response(call_prompt, temp=True), system_prompt=SYSTEM_PROMPT) # Shove historical information into the memory (temporarily)
    response_dict = parse_json_response(response)
        
    # Extract function name and query
    function_name = response_dict['function_name']
    query = response_dict['query']
    
    print("Calling Function: ", function_name, " | Query: ", query)

    # Call the appropriate function based on the response
    if function_name == "search_aor_with_item":
        search_str = memory.search_aor_with_item(query)
        return search_str, memory, False
    elif function_name == "search_aor_with_no":
        search_str = memory.search_aor_with_no(query)
        return search_str, memory, False
    elif function_name == "search_invoice_with_item":
        search_str = memory.search_invoice_with_item(query)
        return search_str, memory, False
    elif function_name == "search_invoice_with_no":
        search_str = memory.search_invoice_with_no(query)
        return search_str, memory, False
    elif function_name == "query_detail":
        return memory.query_detail(user_query), memory, True # ICL answer should take user_query directly
    elif function_name == "query_invoice_detail":
        return memory.query_invoice_detail(user_query), memory, True # ICL answer should take user_query directly
    # elif function_name == "direct_answer": # Direct answer should have reset the memory state ? maybe NOT !
    else:
        print("Direct Answer")
        direct_prompt = DIRECT_ANSWER_TEMPLATE.format(
            invoice_narrative=memory.invoice_narrative, aor_narrative=memory.narrative, user_query=user_query
        )
        response_str = get_oai_response(memory.update_user_response(direct_prompt, temp=True), system_prompt=SYSTEM_PROMPT)
        thought_str, answer_str = parse_thought_answer(response_str)        
        return answer_str, memory, True

def query_memory(query, memory: Memory):
    terminate = False
    i = 0
    memory.last_call = []
    while not terminate:
        info_str, memory, terminate = query_memory_single(query, memory)
        if not terminate:
            print(f"Retrived Information: {info_str}")
        else:
            print(f"Final Answer: {info_str}")
        i += 1
        terminate = terminate or (i>3)
    return info_str, memory

