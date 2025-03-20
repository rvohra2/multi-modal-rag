from pdfprocessor import PDFProcessor
from processor import Processor
from database import QdrantDatabase
from typing import List, Dict, Any

pdfprocessor = PDFProcessor()
processor = Processor()

class VectorSearchService:
    def __init__(self, instance_id:str, create_collection: bool = True):
        """
        Initializes the Middleware with a Qdrant database connection.

        Args:
            collection_id (str): Unique identifier for the collection.
            create_collection (bool, optional): Whether to create a new collection. Defaults to True.
        """
        qdrant_url = None
        self.vector_db_manager  = QdrantDatabase(qdrant_url=None, collection_name="colpali", create_collection=create_collection)

        try:
            # Verify collection initialization
            count_result = self.vector_db_manager.client.count(collection_name="colpali")
            print(f"Collection initialized with {count_result.count} vectors.")
        except Exception as e:
            print(f"Error accessing collection: {e}")
        
    
    def search_text_queries(self, queries: list[str]) -> List[List[tuple]]:
        """
        Searches the Qdrant collection for the given queries.

        Args:
            queries (List[str]): List of text queries to search for.

        Returns:
            List[List[tuple]]: List of search results for each query.
        """
        print(f"Searching for {len(queries)} queries")

        search_results = []
        try:
            for query in queries:
                print(f"Processing query: {query}")
                query_vector  = processor.generate_text_embeddings([query])[0]

                results  = self.vector_db_manager.search_vectors(query_vector , top_k=5)
                search_results.append(results)
            print("Search completed.")
            return search_results
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    

    def index_pdf_images(self, pdf_path: str, instance_id: str, max_pages: int, selected_pages: list[int] = None) -> list[str]:
        """
        Indexes a PDF by extracting images, generating embeddings, and storing them in Qdrant.

        Args:
            pdf_path (str): Path to the PDF file.
            collection_id (str): Unique identifier for the collection.
            max_pages (int): Maximum number of pages to process.
            specific_pages (List[int], optional): Specific pages to process. Defaults to None.

        Returns:
            List[str]: List of file paths to the extracted images.
        """
        # Reset the collection before indexing new data
        self.vector_db_manager.reset_collection()
        print(f"Indexing PDF: {pdf_path}, Instance ID: {instance_id}, Max Pages: {max_pages}")

        try:
            # Extract images from the PDF
            image_files = pdfprocessor.extract_images(instance_id, pdf_path, max_pages)
            print(f"Extracted {len(image_files)} images.")

            # Generate embeddings for the images
            image_vectors = processor.generate_image_embeddings(image_files)
            
            image_data = [
                    {"colbert_vecs": image_vectors[i], "filepath": image_files[i]}
                    for i in range(len(image_files))
                ]

            # Store the image embeddings in Qdrant
            print(f"Storing {len(image_data)} image embeddings in Qdrant...")
            self.vector_db_manager.store_image_vectors(image_data)
            
            # Verify insertion
            count_result = self.vector_db_manager.client.count(collection_name="colpali")
            print(f"Indexing completed. Total vectors stored: {count_result.count}")

            return image_files
        except Exception as e:
            print(f"Error during indexing: {e}")
            return []
