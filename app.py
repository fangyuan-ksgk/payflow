import streamlit as st
from src.agent import RagAgent 
from PIL import Image

agent = RagAgent()

# Logo Image Path
logo_path = "./assets/logo.png"

def main():
    st.set_page_config(layout="wide")

    # Display logo
    logo = Image.open(logo_path)
    st.image(logo, width=200)

    st.title("Admin Agent")

    # Sidebar for invoice and AOR images
    with st.sidebar:
        if agent.memory.invoice_image:
            st.subheader("Invoice")
            st.image(agent.memory.invoice_image, use_column_width=True)
            st.markdown(agent.memory.invoice_narrative)

        if agent.memory.aor_image:
            st.subheader("AOR")
            st.image(agent.memory.aor_image, use_column_width=True)
            st.markdown(agent.memory.narrative)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the message using the RagAgent
        agent_response = agent.chat(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(agent_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": agent_response})

if __name__ == "__main__":
    main()