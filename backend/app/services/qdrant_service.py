from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVector
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.services.bm25_service import bm25_service
from shared.models.schemas import DocumentChunk
import uuid
import asyncio
import time

class QdrantService:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            timeout=120  # Set timeout to 120 seconds
        )
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.batch_size = 1000  # Process chunks in batches of 1000
        self._ensure_collection()
    
    def _ensure_collection(self):
        """Ensure the collection exists with both dense and sparse vector support"""
        try:
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                from qdrant_client.models import VectorParams, SparseVectorParams
                
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "dense": VectorParams(
                            size=settings.QDRANT_VECTOR_SIZE,
                            distance=Distance.COSINE
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams()
                    }
                )
        except Exception as e:
            raise Exception(f"Failed to ensure collection: {str(e)}")
    
    async def add_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Add document chunks to Qdrant in batches with both dense and sparse vectors"""
        try:
            # Fit BM25 model on all chunk texts if not already fitted
            if not bm25_service.bm25_model:
                chunk_texts = [chunk.content for chunk in chunks]
                bm25_service.fit(chunk_texts)
            
            # Convert chunks to points first
            points = []
            for chunk in chunks:
                if chunk.embedding:
                    # Generate sparse vector using BM25
                    sparse_vector_dict = bm25_service.get_sparse_vector(chunk.content)
                    sparse_vector = SparseVector(
                        indices=list(sparse_vector_dict.keys()),
                        values=list(sparse_vector_dict.values())
                    )
                    
                    point = PointStruct(
                        id=str(uuid.uuid4()),
                        vector={
                            "dense": chunk.embedding,
                            "sparse": sparse_vector
                        },
                        payload={
                            "content": chunk.content,
                            "metadata": chunk.metadata,
                            "source_file": chunk.source_file,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "created_at": chunk.created_at.isoformat()
                        }
                    )
                    points.append(point)
            
            if not points:
                return True
            
            # Process in batches to avoid timeout
            total_points = len(points)
            print(f"[QDRANT] Processing {total_points} points in batches of {self.batch_size}")
            
            for i in range(0, total_points, self.batch_size):
                batch = points[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (total_points + self.batch_size - 1) // self.batch_size
                
                print(f"[QDRANT] Upserting batch {batch_num}/{total_batches} ({len(batch)} points)")
                
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        start_time = time.time()
                        await asyncio.to_thread(
                            self.client.upsert,
                            collection_name=self.collection_name,
                            points=batch
                        )
                        end_time = time.time()
                        print(f"[QDRANT] Batch {batch_num} completed in {end_time - start_time:.2f}s")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count < max_retries:
                            wait_time = 2 ** retry_count  # Exponential backoff
                            print(f"[QDRANT] Batch {batch_num} failed (attempt {retry_count}/{max_retries}), retrying in {wait_time}s: {str(e)}")
                            await asyncio.sleep(wait_time)
                        else:
                            raise Exception(f"Failed to upsert batch {batch_num} after {max_retries} attempts: {str(e)}")
                
                # Small delay between batches to avoid overwhelming Qdrant
                if i + self.batch_size < total_points:
                    await asyncio.sleep(0.1)
            
            print(f"[QDRANT] Successfully processed all {total_points} points")
            return True
            
        except Exception as e:
            raise Exception(f"Failed to add chunks: {str(e)}")
    
    async def search_dense(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using dense vectors"""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", query_embedding),
                limit=limit,
                #score_threshold=score_threshold
            )
            
            return [
                {
                    "id": str(hit.id),  # Add the id field from Qdrant point
                    "content": hit.payload["content"],
                    "metadata": hit.payload["metadata"],
                    "source_file": hit.payload["source_file"],
                    "page_number": hit.payload.get("page_number"),
                    "chunk_index": hit.payload["chunk_index"],
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            raise Exception(f"Failed to search similar chunks: {str(e)}")
    
    async def search_shallow(
        self,
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using sparse/shallow vectors (BM25)"""
        try:
            # Generate sparse vector for the query
            sparse_vector_dict = bm25_service.get_sparse_vector(query_text)
            sparse_vector = SparseVector(
                indices=list(sparse_vector_dict.keys()),
                values=list(sparse_vector_dict.values())
            )
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("sparse", sparse_vector),
                limit=limit,
                #score_threshold=score_threshold
            )
            
            return [
                {
                    "id": str(hit.id),
                    "content": hit.payload["content"],
                    "metadata": hit.payload["metadata"],
                    "source_file": hit.payload["source_file"],
                    "page_number": hit.payload.get("page_number"),
                    "chunk_index": hit.payload["chunk_index"],
                    "score": hit.score
                }
                for hit in results
            ]
        except Exception as e:
            raise Exception(f"Failed to search shallow chunks: {str(e)}")
    
    async def search_hybrid(
        self,
        query_embedding: List[float],
        query_text: str,
        limit: int = 5,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining dense and sparse vectors with weighted scores"""
        try:
            # Perform both searches in parallel
            dense_results_task = self.search_dense(query_embedding, limit=limit, score_threshold=0.0)
            sparse_results_task = self.search_shallow(query_text, limit=limit, score_threshold=0.0)
            
            dense_results, sparse_results = await asyncio.gather(dense_results_task, sparse_results_task)
            
            # Create a dictionary to combine results by document ID
            combined_results = {}
            
            # Process dense results
            for result in dense_results:
                doc_id = result["id"]
                combined_results[doc_id] = {
                    **result,
                    "dense_score": result["score"],
                    "sparse_score": 0.0,
                    "hybrid_score": dense_weight * result["score"]
                }
            
            # Process sparse results and combine
            for result in sparse_results:
                doc_id = result["id"]
                if doc_id in combined_results:
                    # Document found in both searches - combine scores
                    combined_results[doc_id]["sparse_score"] = result["score"]
                    combined_results[doc_id]["hybrid_score"] = (
                        dense_weight * combined_results[doc_id]["dense_score"] +
                        sparse_weight * result["score"]
                    )
                else:
                    # Document only found in sparse search
                    combined_results[doc_id] = {
                        **result,
                        "dense_score": 0.0,
                        "sparse_score": result["score"],
                        "hybrid_score": sparse_weight * result["score"]
                    }
            
            # Sort by hybrid score and apply limit and threshold
            final_results = [
                {
                    "id": result["id"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "source_file": result["source_file"],
                    "page_number": result["page_number"],
                    "chunk_index": result["chunk_index"],
                    "score": result["hybrid_score"],
                    "dense_score": result["dense_score"],
                    "sparse_score": result["sparse_score"]
                }
                for result in sorted(
                    combined_results.values(),
                    key=lambda x: x["hybrid_score"],
                    reverse=True
                )
                if result["hybrid_score"] >= score_threshold
            ][:limit]
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Failed to perform hybrid search: {str(e)}")
    
    async def check_health(self) -> bool:
        """Check if Qdrant service is available"""
        try:
            self.client.get_collections()
            return True
        except:
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information"""
        try:
            # Check if collection exists and get basic info
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                return {
                    "name": self.collection_name,
                    "vectors_count": 0,
                    "status": "not_found"
                }
            
            # Collection exists, try to get count using count method instead
            try:
                # Use count_points instead of get_collection to avoid parsing issues
                count_result = self.client.count(
                    collection_name=self.collection_name,
                    exact=True
                )
                vectors_count = count_result.count if hasattr(count_result, 'count') else 0
            except Exception:
                # If count fails, just return 0
                vectors_count = 0
            
            return {
                "name": self.collection_name,
                "vectors_count": vectors_count,
                "status": "active"
            }
            
        except Exception as e:
            print(f"Error getting collection info: {str(e)}")
            # Return safe default
            return {
                "name": self.collection_name,
                "vectors_count": 0,
                "status": "error"
            }

qdrant_service = QdrantService()
