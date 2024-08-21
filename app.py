import streamlit as st
from src.agent import RagAgent 
from PIL import Image

# Initialize RagAgent only once
if 'agent' not in st.session_state:
    st.session_state.agent = RagAgent()

# Logo Image Path
# logo_path = "./assets/red-icon.png"
logo_path = "./assets/Pay-Flow-Blue.png"
# agent_icon_path = "./assets/icon-2.jpeg"
agent_icon_path = "./assets/Assistant.png"
# user_icon_path = "./assets/user-icon.png"
user_icon_path = "./assets/User.png"
document_icon_path = "./assets/document-icon.png"

agent_icon = Image.open(agent_icon_path)
user_icon = Image.open(user_icon_path)
docu_icon = Image.open(document_icon_path)

def main():
    st.set_page_config(layout="wide")

    # Initialize session state for invoice narrative
    if "invoice_narrative" not in st.session_state:
        st.session_state.invoice_narrative = ""

    # Display logo
    col1, col2 = st.columns([1, 4])  # Adjusted column ratio to move content left
    with col1:
        logo = Image.open(logo_path)
        st.image(logo, width=280, use_column_width=True)  # Use column width for responsive sizing
    # with col2:
    #     st.markdown(
    #         "<div style='display: flex; flex-direction: column; justify-content: flex-end; height: 100%;'>"
    #         "<h1 style='font-size: 32px; margin: 0; padding-bottom: 20px;'>Your personal finance assistant</h1>"
    #         "</div>",
    #         unsafe_allow_html=True
    #     )
    # st.markdown("---")  # Draws a horizontal line below the title and image

    # Create two columns: one for the chat interface and one for the sidebar content
    chat_col, sidebar_col = st.columns([3, 1])

    with chat_col:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=user_icon if message["role"] == "user" else agent_icon):
                st.markdown(message["content"])

    # Chat input - outside of any column
    if prompt := st.chat_input("How can I help you?"):
        # Display user message in chat message container
        with chat_col:
            with st.chat_message("user", avatar=user_icon):
                st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the message using the RagAgent
        agent_response = st.session_state.agent.chat(prompt)

        # Display assistant response in chat message container
        with chat_col:
            with st.chat_message("assistant", avatar=agent_icon):
                st.markdown(agent_response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": agent_response})

        # Update the invoice narrative in the session state
        st.session_state.invoice_narrative = st.session_state.agent.memory.invoice_narrative

    with sidebar_col:
        # st.image(docu_icon, width=80)  # Adjusted width to fit the sidebar
        # Sidebar for invoice and AOR images
        if st.session_state.agent.memory.invoice_image:
            st.subheader("Retrieved Invoice")
            st.image(st.session_state.agent.memory.invoice_image, width=300)  # Adjust width as needed
            st.markdown(st.session_state.invoice_narrative)

        # Ensure AOR section is always displayed, regardless of invoice presence
        if st.session_state.agent.memory.aor_image:
            st.subheader("Retrieved AOR")
            st.image(st.session_state.agent.memory.aor_image, width=300)  # Adjust width as needed
            st.markdown(st.session_state.agent.memory.narrative)

if __name__ == "__main__":
    main()