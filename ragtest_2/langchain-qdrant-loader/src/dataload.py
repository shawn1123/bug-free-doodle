from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import os
import pandas as pd
from docx import Document
import pdfplumber
from langchain.embeddings import OpenAIEmbeddings  # Example embedding model from LangChain

def load_data_to_qdrant(file_paths, collection_name, qdrant_host="localhost", qdrant_port=6333):
    """
    Load data from Excel, Word, and PDF files into a Qdrant vector database.

    Args:
        file_paths (list): List of file paths to load.
        collection_name (str): Name of the Qdrant collection.
        qdrant_host (str): Host of the Qdrant server.
        qdrant_port (int): Port of the Qdrant server.
    """
    # Initialize Qdrant client
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    # Ensure the collection exists
    client.recreate_collection(
        collection_name=collection_name,
        vector_size=300,  # Adjust based on your embedding size
        distance="Cosine"
    )

    embeddings = OpenAIEmbeddings()  # Initialize LangChain embeddings
    points = []
    for file_path in file_paths:
        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            # Process Excel files
            xls = pd.ExcelFile(file_path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                for index, row in df.iterrows():
                    text = " ".join(map(str, row.values))
                    vector = embeddings.embed(text)  # Generate vector using LangChain
                    points.append(PointStruct(
                        id=len(points), 
                        vector=vector, 
                        payload={
                            "text": text, 
                            "file_name": os.path.basename(file_path), 
                            "sheet_name": sheet_name, 
                            "row_index": index
                        }
                    ))

        elif file_path.endswith(".docx"):
            # Process Word files
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            vector = embeddings.embed(text)  # Generate vector using LangChain
            points.append(PointStruct(id=len(points), vector=vector, payload={"text": text, "file_name": os.path.basename(file_path)}))

        elif file_path.endswith(".pdf"):
            # Process PDF files
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                vector = embeddings.embed(text)  # Generate vector using LangChain
                points.append(PointStruct(id=len(points), vector=vector, payload={"text": text, "file_name": os.path.basename(file_path)}))

        else:
            print(f"Unsupported file type: {file_path}")

    # Upload points to Qdrant
    client.upsert(collection_name=collection_name, points=points)
    print(f"Data successfully loaded into Qdrant collection '{collection_name}'.")