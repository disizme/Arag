import ollama
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from backend.app.core.config import settings

class OllamaService:
    def __init__(self):
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    
    async def get_embedding(self, text: str, model: str = None) -> List[float]:
        """Get embedding for a text using Ollama"""
        try:
            embedding_model = model or settings.OLLAMA_EMBEDDING_MODEL
            response = await asyncio.to_thread(
                self.client.embeddings,
                model=embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")
    
    async def generate_response(self, query: str, context: str, model_name: str) -> str:
        """Generate response using Ollama with context"""
        try:
            prompt = f"""Context: {context}
            
Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so clearly."""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=model_name,
                prompt=prompt
            )
            return response['response']
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    async def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = await asyncio.to_thread(self.client.list)
            return [model['name'] for model in response['models']]
        except Exception as e:
            raise Exception(f"Failed to list models: {str(e)}")
    
    async def check_health(self) -> bool:
        """Check if Ollama service is available"""
        try:
            await asyncio.to_thread(self.client.list)
            return True
        except:
            return False
    
    async def multi_step_reasoning(
        self, 
        query: str, 
        context_retrieval_func,
        model_name: str,
        max_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Perform multi-step reasoning for complex queries using retrieved contexts
        
        This method breaks down complex queries into sub-questions, retrieves
        relevant context for each step, and builds a reasoning chain to arrive
        at a comprehensive answer.
        
        Args:
            query: The complex query to answer
            context_retrieval_func: Function to retrieve context for sub-queries
            model_name: The LLM model to use
            max_steps: Maximum number of reasoning steps
            
        Returns:
            Dict containing the final answer, reasoning steps, and metadata
        """
        try:
            reasoning_steps = []
            accumulated_knowledge = ""
            
            # Step 1: Decompose the complex query into sub-questions
            sub_questions = await self._decompose_query(query, model_name)
            
            # Limit the number of sub-questions to max_steps
            sub_questions = sub_questions[:max_steps]
            
            # Step 2: Process each sub-question
            for i, sub_question in enumerate(sub_questions):
                step_num = i + 1
                
                # Retrieve context for this sub-question
                sub_context = await context_retrieval_func(sub_question)
                
                # Generate answer for this step using context + accumulated knowledge
                step_answer = await self._reason_step(
                    sub_question, 
                    sub_context, 
                    accumulated_knowledge,
                    model_name
                )
                
                # Store this reasoning step
                step_info = {
                    "step_number": step_num,
                    "sub_question": sub_question,
                    "context_used": sub_context,
                    "step_answer": step_answer
                }
                reasoning_steps.append(step_info)
                
                # Accumulate knowledge for next steps
                accumulated_knowledge += f"\nStep {step_num}: {sub_question}\nAnswer: {step_answer}\n"
            
            # Step 3: Synthesize final answer from all reasoning steps
            final_answer = await self._synthesize_final_answer(
                query, 
                reasoning_steps, 
                accumulated_knowledge,
                model_name
            )
            
            return {
                "query": query,
                "final_answer": final_answer,
                "reasoning_steps": reasoning_steps,
                "num_steps": len(reasoning_steps),
                "model_used": model_name
            }
            
        except Exception as e:
            raise Exception(f"Failed in multi-step reasoning: {str(e)}")
    
    async def _decompose_query(self, query: str, model_name: str) -> List[str]:
        """Break down a complex query into sub-questions"""
        try:
            decomposition_prompt = f"""Given the following complex question, break it down into 2-3 simpler sub-questions that need to be answered to fully address the main question.

Main Question: {query}

Please provide only the sub-questions, one per line, without numbering or additional text. Focus on questions that require specific information retrieval.

Sub-questions:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=model_name,
                prompt=decomposition_prompt
            )
            
            # Parse the response to extract sub-questions
            sub_questions = [
                q.strip() 
                for q in response['response'].split('\n') 
                if q.strip() and not q.strip().startswith('#')
            ]
            
            # Filter out empty or very short questions
            sub_questions = [q for q in sub_questions if len(q) > 10]
            
            return sub_questions[:3]  # Limit to 3 sub-questions
            
        except Exception as e:
            # Fallback: return the original query as a single step
            return [query]
    
    async def _reason_step(
        self, 
        sub_question: str, 
        context: str, 
        accumulated_knowledge: str,
        model_name: str
    ) -> str:
        """Reason through a single step using context and accumulated knowledge"""
        try:
            step_prompt = f"""You are answering a sub-question as part of a multi-step reasoning process.

Previous Knowledge (from earlier steps):
{accumulated_knowledge if accumulated_knowledge else "None"}

Current Context:
{context}

Sub-question: {sub_question}

Please provide a clear, concise answer to this sub-question based on the context and any relevant previous knowledge. Focus on factual information that will help answer the broader question."""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=model_name,
                prompt=step_prompt
            )
            
            return response['response']
            
        except Exception as e:
            return f"Error in reasoning step: {str(e)}"
    
    async def _synthesize_final_answer(
        self, 
        original_query: str, 
        reasoning_steps: List[Dict], 
        accumulated_knowledge: str,
        model_name: str
    ) -> str:
        """Synthesize a final comprehensive answer from all reasoning steps"""
        try:
            # Format the reasoning chain
            steps_summary = ""
            for step in reasoning_steps:
                steps_summary += f"""
Step {step['step_number']}: {step['sub_question']}
Answer: {step['step_answer']}
"""
            
            synthesis_prompt = f"""You have completed a multi-step reasoning process to answer a complex question. Now synthesize a comprehensive final answer.

Original Question: {original_query}

Reasoning Steps Completed:
{steps_summary}

Based on all the reasoning steps above, provide a comprehensive, well-structured answer to the original question. 
- Integrate insights from all steps
- Ensure logical flow and coherence
- Address the question completely
- Be concise but thorough

Final Answer:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=model_name,
                prompt=synthesis_prompt
            )
            
            return response['response']
            
        except Exception as e:
            # Fallback: concatenate step answers
            fallback_answer = "Based on the analysis:\n"
            for i, step in enumerate(reasoning_steps):
                fallback_answer += f"{i+1}. {step['step_answer']}\n"
            return fallback_answer

ollama_service = OllamaService()