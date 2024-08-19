import chainlit as cl
from src.agent import RagAgent 
agent = RagAgent()

@cl.step(type="tool")
async def tool():
    # Fake tool
    await cl.sleep(2)
    return "Response from the tool!"


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It processes the user's message using the RagAgent and sends the response back.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    # Get the user's message content
    user_message = message.content

    # Process the message using the RagAgent
    agent_response = agent.chat(user_message)

    # Send the agent's response back to the user
    response_message = cl.Message(content=agent_response)
    await response_message.send()