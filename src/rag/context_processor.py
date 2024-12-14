from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from langchain_core.language_models.llms import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessedChunk(BaseModel):
    """Represents a processed document chunk with compression"""
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Document) -> "ProcessedChunk":
        return cls(
            content=document.page_content,
            metadata=document.metadata
        )

    def to_document(self) -> Document:
        return Document(
            page_content=self.content,
            metadata=self.metadata
        )

class ContextProcessor:
    def __init__(self,
                 llm: Optional[Union[BaseLLM, BaseChatModel]] = None,
                 max_tokens: int = 2000,
                 model_path: Optional[str] = None):
        """
        Initialize the context processor with either a provided LLM or create a default one
        
        Args:
            llm: Optional pre-configured LLM instance
            max_tokens: Maximum tokens for compressed context
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
            self.llm = ChatOpenAI(model="gpt-3.5-turbo")
            
        self.max_tokens = max_tokens
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
                    metadata=chunk.metadata
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
        """
        Process and compress documents
        
        Args:
            documents: List of Document objects or ProcessedChunks
            query: Original query string (kept for interface compatibility)
            
        Returns:
            List of processed Document objects
        """
        try:
            chunks = self._validate_input(documents)
            compressed_chunks = self.compress_chunks(chunks)
            return [chunk.to_document() for chunk in compressed_chunks]

        except Exception as e:
            logger.error(f"Error in document processing pipeline: {e}")
            return documents if isinstance(documents[0], Document) else [c.to_document() for c in documents]