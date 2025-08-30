from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVector, Prefetch, FusionQuery, Fusion
from typing import List, Dict, Any, Optional
from backend.app.core.config import settings
from backend.app.services.bge_service import bge_service
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
        """Add document chunks to Qdrant in batches with both dense and sparse vectors using BGE-M3"""
        try:            
            # Convert chunks to points with BGE-M3 embeddings
            points = []
            for chunk in chunks:
                # Generate both dense and sparse embeddings using BGE-M3
                embeddings = await bge_service.get_embeddings(chunk.content)
                
                sparse_vector = SparseVector(
                    indices=list(embeddings.sparse.keys()),
                    values=list(embeddings.sparse.values())
                )
                
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": embeddings.dense,
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
        query_text: str, 
        limit: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks using dense vectors (BGE-M3)"""
        try:
            # Generate dense embedding for the query using BGE-M3
            embeddings = await bge_service.get_embeddings(query_text)
            
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=("dense", embeddings.dense),
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
        """Search for similar chunks using sparse/shallow vectors (BGE-M3)"""
        try:
            # Generate sparse vector for the query using BGE-M3
            sparse_vector_dict = await bge_service.get_sparse_vector(query_text)
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
        query_text: str,
        limit: int = 5,
        score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Native hybrid search using Qdrant's built-in RRF fusion with BGE-M3"""
        try:
            # Get both dense and sparse embeddings from BGE-M3
            embeddings = await bge_service.get_embeddings(query_text)
            
            sparse_vector = SparseVector(
                indices=list(embeddings.sparse.keys()),
                values=list(embeddings.sparse.values())
            )
            
            # Use Qdrant's native hybrid search with RRF fusion
            results = await asyncio.to_thread(
                self.client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=sparse_vector,
                        using="sparse",
                        limit=limit * 2  # Get more candidates for better fusion
                    ),
                    Prefetch(
                        query=embeddings.dense,
                        using="dense",
                        limit=limit * 2  # Get more candidates for better fusion
                    )
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit
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
                for hit in results.points
                if hit.score >= score_threshold
            ]
            
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
