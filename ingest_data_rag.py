# ingest_data_rag.py
# This file handles the ingestion of data for a Retrieval-Augmented Generation (RAG) system
# It prepares documents from a CSV file and URLs, and ingests them into a vector database
# It also queries the vector database to demonstrate the ingestion process
import os
import sys
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient, RAGDocument
import pandas as pd
import logging
from constants import LOG_LEVELS

# Class to handle data ingestion for RAG (Retrieval-Augmented Generation)
# This class prepares documents from a CSV file and URLs, and ingests them into a vector database
# It also queries the vector database to demonstrate the ingestion process  
class IngestDataRAG:
    
    # Function to initialize the IngestDataRAG class
    def __init__(self):
        """Initialize the AgentInfra with LlamaStackClient and model parameters."""

        sys.path.append('..')
        # Load environment variables from .env file
        load_dotenv()

        root_log_level = os.getenv("ROOT_LOG_LEVEL", "INFO")
        app_log_level = os.getenv("APP_LOG_LEVEL", "INFO")

        # Set logging levels
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.basicConfig(level=LOG_LEVELS[root_log_level], format='%(asctime)s - %(levelname)s - %(message)s', force=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(LOG_LEVELS[app_log_level])
        
        # Initialize the Llama Stack client
        self.client = LlamaStackClient(
            base_url=os.getenv("LLAMA_STACK_SERVER_URL", "http://localhost:8321")
        )

        self.logger.info("Connected to Llama Stack server")

    # Function to prepare documents from a CSV file
    # Each row in the CSV is converted into a Llama Stack `Document` object
    def prepare_documents_from_csv(self, file_path):
        """
        Reads a CSV file and converts each row into a Llama Stack `Document` object.
        """

        self.logger.info(f"Preparing documents from {file_path}...")
        
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)

        # Get the list of columns in the DataFrame
        documents = []
        for index, row in df.iterrows():
            # Combine relevant columns into the document's content/text
            # This is what the embedding model will primarily "read"
            text_content = (
                f"Product Name: {row['Product']}. SKU Description: {row['SKU_Description']}. "
                f"SKU Number: {row['SKU']}. Price: ${row['List_Price']}. "
            )

            # Include all original CSV columns as metadata
            # This metadata can be used for filtering during retrieval or just for context
            metadata = row.to_dict()

            self.logger.debug(f"Processing document {index + 1}: {text_content}...")

            # Create document object to ingest
            documents.append(
                RAGDocument(
                    document_id=str(index) + "-SKU-RH-LATAM-Q3-2025",  # Use the index or a unique identifier
                    mime_type="text/plain",  # Assuming the content is plain text
                    content=text_content, # 'content' is the field for the main text
                    metadata=metadata
                )
            )
        
        self.logger.info(f"Prepared {len(documents)} documents.")
        return documents
    
    # Function to prepare documents from a list of URLs
    # Each URL is converted into a Llama Stack `Document` object
    def prepare_documents_from_urls(self):
        """
        Converts a list of URLs into Llama Stack `Document` objects.
        """

        self.logger.info("Preparing documents from URLs...")

        # ingest the documents into the newly created document collection
        urls = [
            #("https://raw.githubusercontent.com/lcoronad/ai-transformers/main/skus_red_hat_openshift.rst", "text/plain"),
            ("https://www.openshift.guide/openshift-guide-screen.pdf", "application/pdf"),
        ]

        # Create document object to ingest
        documents = [
            RAGDocument(
                document_id=f"num-{i}",
                content=url,
                mime_type=url_type,
                metadata={},
            )
            for i, (url, url_type) in enumerate(urls)
        ]

        self.logger.info(f"Prepared {len(documents)} documents from URLs.")
        return documents

    # Function to ingest data into the RAG system
    # It registers a vector database, prepares documents from a CSV file, and ingests them
    def ingest_data(self, vector_db_id, documents) -> None:
        """
        Ingest data into the RAG (Retrieval-Augmented Generation) system.
        """

        # define and register the document collection to be used
        self.client.vector_dbs.register(
            vector_db_id=vector_db_id,
            embedding_model=os.getenv("VDB_EMBEDDING"),
            embedding_dimension=int(os.getenv("VDB_EMBEDDING_DIMENSION", 384)),
            provider_id=os.getenv("VDB_PROVIDER"),
        )
        
        # Insert documents to the vector database
        self.client.tool_runtime.rag_tool.insert(
            documents=documents,
            vector_db_id=vector_db_id,
            chunk_size_in_tokens=int(os.getenv("VECTOR_DB_CHUNK_SIZE", 512)),
        )

        self.logger.info(f"Documents ingested into RAG {vector_db_id} successfully.")
    
    # Function to query the RAG system
    # It retrieves documents based on a specific query
    def query_rag(self, vector_db_id, query):
        """
        Query the RAG system to retrieve documents based on a specific query.
        """

        # Execute the query against the vector database
        result = self.client.tool_runtime.rag_tool.query(
            vector_db_ids=[vector_db_id], 
            query_config={"query": query}, 
            content=query
        )

        self.logger.info(f"RAG Query from {vector_db_id} - Result: \n{result.content}")
    
    # Process the data, preparing documents from a CSV file and ingesting them into the RAG system
    def process_data(self):
        """
        Process data by preparing documents from a CSV file and ingesting them into the RAG system.
        """

        self.logger.info("Starting data processing...")
        vector_db_id = "skus_rh_vector_db"
        # Prepare documents from the CSV file
        documents = self.prepare_documents_from_csv("data/Commercial-Direct-LATAM-USD-Q3-2025-Subscriptions.csv")
        # Ingest data into the RAG system
        self.ingest_data(vector_db_id, documents)
        # Query the RAG system
        self.query_rag(vector_db_id, "List of Red Hat OpenShift SKUs")

        vector_db_id = "ocp_rh_vector_db"
        # Prepare documents from the CSV file
        documents = self.prepare_documents_from_urls()
        # Ingest data into the RAG system
        self.ingest_data(vector_db_id, documents)
        # Query the RAG system
        self.query_rag(vector_db_id, "What is Red Hat OpenShift?")

# Launch the interface and MCP server
if __name__ == "__main__":
    # Create an instance of the IngestDataRAG class
    ingest_data_rag = IngestDataRAG()
    # Process the data
    ingest_data_rag.process_data()
    print("Data ingestion and querying completed successfully.")
    