import streamlit as st
from src.helper import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain
)


# 🔹 Handle user question
def user_input(user_question):

    if st.session_state.conversation is None:
        st.error("⚠️ Please upload and process PDFs first.")
        return

    response = st.session_state.conversation.invoke(
        {"question": user_question}
    )

    st.session_state.chatHistory = response["chat_history"]

    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("👤 User:", message.content)
        else:
            st.write("🤖 Bot:", message.content)


def main():
    st.set_page_config(page_title="Information Retrieval System")
    st.header("📄 Chat with your PDFs")

    # 🔹 Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    # 🔹 User input
    user_question = st.text_input("Ask a question from your PDFs")

    if user_question and st.session_state.conversation:
        user_input(user_question)

    # 🔹 Sidebar
    with st.sidebar:
        st.title("📂 Menu")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)

                try:
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.success("✅ Processing complete!")
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()