'''
Derived from VLM2Vec (https://github.com/TIGER-AI-Lab/VLM2Vec)
MIT License
'''
import numpy as np
import hnswlib
from src.utils import print_rank


class HNSWIndex:
    """
    Manages HNSW indices for different candidate types and their associated keys.
    This implementation provides functionality similar to FAISSIndex.
    """
    def __init__(self, ef_construction=200, M=48):
        self.indices = {}  # Stores HNSW indices for each candidate type
        self.keys_dict = {}  # Stores candidate keys for each candidate type
        self.dimensions = {}  # Stores embedding dimensions for each candidate type
        self.ef_construction = ef_construction  # Controls index quality
        self.M = M  # Controls graph connectivity
        print_rank(f"HNSW Index initialized with ef_construction={ef_construction}, M={M}")
        
    def create_index(self, cand_type, cand_vectors, cand_keys):
        """
        Create an HNSW index for a candidate type.
        
        Args:
            cand_type (str): Candidate type (state, trajectory, interval)
            cand_vectors (np.ndarray): Embeddings for the candidates
            cand_keys (list): List of candidate IDs
        """
        print_rank(f"Building HNSW index for {cand_type}")
        assert len(cand_keys) == cand_vectors.shape[0]
        # Store candidate keys for this type
        self.keys_dict[cand_type] = cand_keys
        
        # Normalize vectors for cosine similarity
        vectors = cand_vectors.astype(np.float32).copy()
        # Equivalent to faiss.normalize_L2
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        assert not np.any(norms == 0), "Zero norm found in candidate vectors"
        vectors = vectors / norms
        
        num_elements, dim = vectors.shape
        
        # Initialize the index; using cosine metric (distance = 1 - cosine similarity)
        index = hnswlib.Index(space='cosine', dim=dim)
        index.init_index(max_elements=num_elements, ef_construction=self.ef_construction, M=self.M)
        
        # Add all vectors with their IDs
        index.add_items(vectors, np.arange(num_elements))
        
        # Set search quality parameter
        index.set_ef(100)
        
        # Store the index
        self.indices[cand_type] = index
        
    def search(self, cand_type, query_vectors, k):
        """
        Search for nearest neighbors in the index for a specific candidate type.
        
        Args:
            cand_type (str): Candidate type (state, trajectory, interval)
            query_vector (np.ndarray): Query embedding(s)
            k (int): Number of results to retrieve
            
        Returns:
            tuple: (scores, predictions) where:
                - scores is a list of lists of similarity scores
                - predictions is a list of lists of candidate IDs
        """
        if cand_type not in self.indices:
            raise ValueError(f"Index for {cand_type} not found")
        
        if len(query_vectors.shape) == 1:
            q = query_vectors.reshape(1, -1).astype(np.float32)
        else:
            q = query_vectors.astype(np.float32)

        # Normalize query vectors
        norms = np.linalg.norm(q, axis=1, keepdims=True)
        assert not np.any(norms == 0), "Zero norm found in query vectors"
        q = q / norms

        assert q.shape[1] == self.indices[cand_type].dim, \
            f"Query dimension {q.shape[1]} doesn't match index dimension {self.indices[cand_type].dim}"
    
        # Search in the HNSW index
        indices, distances = self.indices[cand_type].knn_query(q, k=k)
        
        # Convert distances to similarity scores
        scores = 1 - distances
        
        # Process results - create a list of predictions for each query
        all_predictions = []
        for i in range(indices.shape[0]):
            predictions = [self.keys_dict[cand_type][int(idx)] for idx in indices[i]]
            all_predictions.append(predictions)
        
        return scores.tolist(), all_predictions
