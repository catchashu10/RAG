# Dependencies
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedChunk(BaseModel):
    """Represents a processed document chunk with relevance scoring and compression"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    relevance_score: float = 0.0

    @classmethod
    def from_document(cls, document: Document) -> "ProcessedChunk":
        return cls(
            content=document.page_content,
            metadata=document.metadata
        )

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata={
                **self.metadata,
                'relevance_score': self.relevance_score
            }
        )

class ContextProcessor:
    def __init__(self,
                 llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
                 max_tokens: int = 2000,
                 min_relevance_score: float = 0.3,
                 model_path: Optional[str] = None):
        """
        Initialize the context processor with either a provided LLM or create a default one
        
        Args:
            llm: Optional pre-configured LLM instance
            max_tokens: Maximum tokens for compressed context
            min_relevance_score: Minimum relevance score threshold
            model_path: Path to local LLaMA model, if using local model
        """
        if llm is not None:
            self.llm = llm
        elif model_path is not None:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=0.1,
                max_tokens=2000,
                n_ctx=2048,
                top_p=0.95,
                verbose=False
            )
        else:
            # Default to OpenAI if no local model specified
            self.llm = ChatOpenAI(model="gpt-3.5-turbo")
            
        self.max_tokens = max_tokens
        self.min_relevance_score = min_relevance_score
        self._is_chat_model = isinstance(self.llm, BaseChatModel)

    def _get_llm_response(self, prompt: str) -> str:
        """Helper method to handle different LLM types"""
        try:
            if self._is_chat_model:
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content.strip()
            else:
                response = self.llm.invoke(prompt)
                return response.strip()
        except Exception as e:
            logger.error(f"Error getting LLM response: {e}")
            raise

    def _validate_input(self, documents: Union[List[Document], List[ProcessedChunk]]) -> List[ProcessedChunk]:
        if not documents:
            logger.warning("No documents provided for processing")
            return []

        chunks = []
        for doc in documents:
            if isinstance(doc, Document):
                chunks.append(ProcessedChunk.from_document(doc))
            elif isinstance(doc, ProcessedChunk):
                chunks.append(doc)
            else:
                raise TypeError(f"Unsupported document type: {type(doc)}")
        return chunks

    def rerank_chunks(self, chunks: List[ProcessedChunk], query: str) -> List[ProcessedChunk]:
        if not chunks:
            return []

        rerank_prompt = f"""Rate how relevant this document is to the query: "{query}"
        Rate relevance from 0-1, where 1 is highly relevant and 0 is irrelevant.
        Return only the numerical score."""

        for chunk in chunks:
            try:
                response = self._get_llm_response(f"{rerank_prompt}\n\nDocument: {chunk.content}")
                # Extract float from response, handling potential formatting issues
                score_str = ''.join(c for c in response if c.isdigit() or c == '.')
                chunk.relevance_score = float(score_str) if score_str else 0.0
            except Exception as e:
                logger.error(f"Error scoring chunk: {e}")
                chunk.relevance_score = 0.0

        chunks = [c for c in chunks if c.relevance_score >= self.min_relevance_score]
        return sorted(chunks, key=lambda x: x.relevance_score, reverse=True)

    def compress_chunks(self, chunks: List[ProcessedChunk]) -> List[ProcessedChunk]:
        if not chunks:
            return []

        compress_prompt = """Compress the following text while preserving key information:

        {text}

        Compressed version:"""

        compressed_chunks = []
        total_tokens = 0

        for chunk in chunks:
            if total_tokens >= self.max_tokens:
                break

            try:
                response = self._get_llm_response(compress_prompt.format(text=chunk.content))
                compressed_chunk = ProcessedChunk(
                    content=response,
                    metadata=chunk.metadata,
                    relevance_score=chunk.relevance_score
                )
                compressed_chunks.append(compressed_chunk)
                total_tokens += len(response.split())
            except Exception as e:
                logger.error(f"Error compressing chunk: {e}")
                compressed_chunks.append(chunk)

        return compressed_chunks

    def process_documents(self,
                         documents: Union[List[Document], List[ProcessedChunk]],
                         query: str) -> List[Document]:
        try:
            chunks = self._validate_input(documents)
            reranked_chunks = self.rerank_chunks(chunks, query)
            compressed_chunks = self.compress_chunks(reranked_chunks)
            return [chunk.to_document() for chunk in compressed_chunks]

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return documents if isinstance(documents[0], Document) else [c.to_document() for c in documents]

def integrate_with_graph(graph_builder, llm: Optional[Union[BaseLLM, BaseChatModel]] = None, model_path: Optional[str] = None):
    """Helper function to integrate with the query analysis graph"""

    def context_processing_agent(state: Dict[str, Any]) -> Dict[str, Any]:
        processor = ContextProcessor(llm=llm, model_path=model_path)
        if 'retrieved_documents' in state and state['retrieved_documents']:
            try:
                processed_docs = processor.process_documents(
                    documents=state['retrieved_documents'],
                    query=state['query']
                )
                return {'processed_documents': processed_docs}
            except Exception as e:
                logger.error(f"Error in context processing agent: {e}")
                return {'processed_documents': state['retrieved_documents']}
        return {'processed_documents': []}

    # Add the node and edges
    graph_builder.add_node("context_processor", context_processing_agent)
    graph_builder.add_edge("retrieval_agent", "context_processor")
    graph_builder.add_edge("context_processor", "combine_queries")

# Export the classes and functions
__all__ = ['ContextProcessor', 'ProcessedChunk', 'integrate_with_graph']