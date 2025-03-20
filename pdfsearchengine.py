import logging
from vectorsearchservice import VectorSearchService
from multimodalrag import MultiModalRAG
from utils import get_or_create_uuid
from io import BytesIO

import tempfile
import shutil
logging.basicConfig(level=logging.INFO)

multimodalrag = MultiModalRAG()

class PDFSearchEngine:
    def __init__(self):
        """
        Initializes the PDF search engine and document storage.
        """
        self.indexed_pdfs = {}  # Tracks indexed PDFs
        self.current_pdf = None
        self.vector_search_engine = VectorSearchService(id, create_collection=True)
        self.file_id = get_or_create_uuid({"user_uuid": None})
        
    def process_pdf_upload(self, uploaded_file, max_pages: int) -> str:
        """
        Uploads and processes a PDF file, extracting and indexing its pages.

        Args:
            session_state (dict): The current state of the application.
            uploaded_file (BytesIO): The uploaded file as a BytesIO object.
            max_pages (int): Maximum number of pages to extract and index.

        Returns:
            str: Status message indicating success or failure.
        """
        print('uploaded_file: ', uploaded_file)
        

        if not uploaded_file:
            return "Error: No file uploaded."

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(uploaded_file.read())
            temp_pdf_path = temp_pdf.name
        print(f"Uploading file: {temp_pdf_path}, ID: {self.file_id}")

        try:
            self.current_pdf = temp_pdf_path

            # # Save the BytesIO object to a temporary file
            # with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            #     shutil.copyfileobj(uploaded_file, tmp_file)
            #     tmp_file_path = tmp_file.name

            # Index the PDF and extract pages
            pages = self.vector_search_engine.index_pdf_images(
                pdf_path=self.current_pdf, 
                instance_id=self.file_id, 
                max_pages=max_pages
            )
            self.indexed_pdfs[self.file_id] = True  # Mark document as indexed

            return f"Success: Uploaded and indexed {len(pages)} pages."
        
        except Exception as e:
            return f"Error processing PDF: {str(e)}"
    
    
    def query_pdf(self, search_query: str, num_results: int = 5) -> tuple:
        """
        Searches the indexed PDFs for the given query and returns relevant results.

        Args:
            session_state (dict): The current state of the application.
            search_query (str): The search query.
            num_results (int, optional): Number of results to return. Defaults to 5.

        Returns:
            tuple: A tuple containing:
                - List of image paths for the top results.
                - The RAG-generated answer.
        """
        if not search_query:
            return "Error: Please enter a search query.", "--"

        # file_id = get_or_create_uuid(session_state)
        print(f"Searching for query: {search_query}")
            
        try:
            # Perform the search
            search_results = self.vector_search_engine.search_text_queries([search_query])[0]

            # Adjust the number of results if fewer are found
            if len(search_results) < num_results:
                print(f"Warning: Only {len(search_results)} results found (requested {num_results}).")
                num_results = len(search_results)

             # Construct image paths for the top results
            img_paths = [
                f"pdf_images/{self.file_id}/image_{result[1] + 1}.png"  # Adjust for 0-based indexing
                for result in search_results[:num_results]
            ]
            print(f"Retrieved image paths: {img_paths}")

            # Generate a RAG response using Gemini
            rag_response = multimodalrag.query_gemini(search_query, img_paths)

            return img_paths, rag_response

        except Exception as e:
            return f"Error during search: {str(e)}", "--"