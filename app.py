import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import openai
from io import StringIO
import os
from document_processing import split_and_embed_documents
# from adversarial_agent import ask_adversarial_questions
from chromadb.utils import embedding_functions


# Set OpenAI key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize ChromaDB client
# chroma_client = chromadb.Client()

def main():
    st.title("Document Evaluator for RAG Applications")

    # Input fields for documents and intent
    uploaded_files = st.file_uploader("Upload Document(s)", accept_multiple_files=True, type=['txt', 'pdf'])
    print('UPLOADED!')
    app_intent = st.text_area("Explain the intent behind the app", "")

    if st.button("Evaluate"):
        if uploaded_files and app_intent:
            st.write("Processing documents and initializing the vector store...")

            # Load documents into memory
            documents = []
            for uploaded_file in uploaded_files:
                stringio = StringIO(uploaded_file.getvalue().decode("latin-1"))
                document_text = stringio.read()
                documents.append(document_text)

            print(documents[0])

            # Initialize ChromaDB collection
            collection_name = "document_collection"
            try:
                db_collection = chroma_client.create_collection(name=collection_name)
            except:
                print('Bullshit issue is here')
            
            # Split and embed documents
            split_and_embed_documents(documents, db_collection)
            st.write("Documents successfully uploaded and indexed.")
            
            # Adversarial process
            st.write("Running adversarial agent...")
            adversarial_results = ask_adversarial_questions(app_intent, db_collection)

            # Display questions and similarity scores
            for result in adversarial_results:
                st.write(f"Question: {result['question']}")
                st.write(f"Similarity score: {result['similarity']}\n")
        else:
            st.error("Please upload documents and provide an intent.")

if __name__ == "__main__":
    main()
