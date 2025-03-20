from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any

class QdrantDatabase:
    def __init__(self, qdrant_url: str, collection_name: str, create_collection: bool = False, vector_dim: int = 128):
        """
        Initializes the QdrantManager with the specified collection and configuration.

        Args:
            qdrant_url (str): URL of the Qdrant server.
            collection_name (str): Name of the collection to manage.
            create_collection (bool, optional): Whether to create the collection if it doesn't exist. Defaults to False.
            dim (int, optional): Dimension of the vectors. Defaults to 128.
        """
        self.client = QdrantClient(qdrant_url if qdrant_url else ":memory:") # Using in-memory Qdrant for testing
        self.collection_name = collection_name
        self.vector_dim  = vector_dim 

        if create_collection:
            self.create_collection()

    def create_collection(self) -> None:
        """
        Creates a new collection if it doesn't already exist.
        """
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:  # Only create if it doesn't exist
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_dim,  # Dimension of the vectors
                        distance=models.Distance.COSINE,  # Distance metric (can be COSINE, DOT, or EUCLID)
                        multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                        ),
                    ),
                )
                print(f"Collection '{self.collection_name}' created.")
        except Exception as e:
            print(f"Error creating collection: {e}")

    def reset_collection(self) -> None:
        """
        Clears all points in the collection and recreates it.
        """
        try:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' cleared.")
            self.create_collection()  # Recreate the collection after clearing
        except Exception as e:
            print(f"Error clearing collection: {e}")

    def search_vectors(self, query_vector, top_k):
        """
        Searches the collection for the top-k most similar vectors.

        Args:
            query_vector (List[float]): The query vector to search with.
            top_k (int): Number of top results to return.

        Returns:
            List[tuple]: List of tuples containing (score, seq_id) for each result.
        """
        try:
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_vectors=True,
                with_payload=True,
            ).points
            return [(hit.score, hit.payload["seq_id"]) for hit in search_results]
        except Exception as e:
            print(f"Error during search: {e}")
            return []
        
    def prepare_image_metadata(self, images_with_vectors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepares image data for insertion into Qdrant.

        Args:
            images_with_vectors (List[Dict[str, Any]]): List of dictionaries containing image data and vectors.

        Returns:
            List[Dict[str, Any]]: List of prepared image data.
        """
        return [
            {
                "colbert_vecs": img["colbert_vecs"],
                "doc_id": idx,
                "filepath": img["filepath"],
            }
            for idx, img in enumerate(images_with_vectors)
        ]

    def store_image_vectors(self, image_data: List[Dict[str, Any]]):
        """
        Inserts image data into the Qdrant collection.

        Args:
            image_data (List[Dict[str, Any]]): List of dictionaries containing image data and vectors.
        """
        try:
            data  = self.prepare_image_metadata(image_data)

            points = [
                models.PointStruct(
                    id=entry["doc_id"],
                    vector=entry["colbert_vecs"],
                    payload={
                        "seq_id": entry["doc_id"],
                        "doc": entry["filepath"],
                    },
                )
                for entry in data
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            print(f"Inserted {len(points)} vectors into '{self.collection_name}'.")

            count_result = self.client.count(collection_name=self.collection_name)
            print(f"Total vectors in collection: {count_result.count}")
        except Exception as e:
            print("Error inserting vectors:", e)

            