from chromadb.utils import embedding_functions

def split_and_embed_documents(documents, db_collection):
    # Split documents into chunks and add them to the ChromaDB collection
    embedding_model = embedding_functions.DefaultEmbeddingFunction()
    
    for doc in documents:
        # For simplicity, we’ll treat each line in the document as a chunk
        chunks = doc.split('\n')
        
        for idx, chunk in enumerate(chunks):
            if chunk.strip():  # Ensure we don't add empty lines
                # Get the embedding for the chunk of text
                embedding = embedding_model.get_embedding(chunk)
                
                # Add the chunk to the vector store with its embedding
                db_collection.add(
                    documents=[chunk],
                    metadatas=[{"source": f"doc_{idx}"}],
                    ids=[f"doc_{idx}"],
                    embeddings=[embedding]
                )