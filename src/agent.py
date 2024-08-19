from .search import query_memory, Memory

class RagAgent:
    def __init__(self):
        print("------- Initializing Agent --------")
        self.memory = Memory()

    def chat(self, user_message):
        agent_response, self.memory = query_memory(user_message, self.memory)
        self.memory.update_user_response(user_message)
        self.memory.update_agent_response(agent_response)
        return agent_response