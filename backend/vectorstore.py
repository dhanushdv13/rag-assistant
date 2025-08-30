from pathlib import Path
from typing import Sequence, List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import settings

# ============================================================================
# Document Loaders
# ============================================================================

def load_pdfs(paths: Sequence[Path]) -> List[Document]:
    """Load PDF documents from given paths"""
    docs = []
    
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf package not found, please install it with 'pip install pypdf'")
    
    for path in paths:
        print(f"📄 Loading PDF: {path}")
        try:
            reader = PdfReader(str(path))
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            
            if text.strip():
                docs.append(Document(
                    page_content=text.strip(), 
                    metadata={"source": str(path), "pages": len(reader.pages)}
                ))
                print(f"✅ Extracted text from {len(reader.pages)} pages")
            else:
                print(f"⚠️ Warning: No text extracted from {path}")
                
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    
    print(f"📚 Loaded {len(docs)} PDF documents")
    return docs

def load_texts(paths: Sequence[Path]) -> List[Document]:
    """Load text documents from given paths"""
    docs = []
    
    for path in paths:
        print(f"📝 Loading text file: {path}")
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                content = f.read()
            
            if content.strip():
                docs.append(Document(
                    page_content=content.strip(), 
                    metadata={"source": str(path), "type": "text"}
                ))
                print(f"✅ Loaded {len(content)} characters")
            else:
                print(f"⚠️ Warning: Empty content in {path}")
                
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    
    print(f"📄 Loaded {len(docs)} text documents")
    return docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    """Split documents into optimized chunks"""
    if not docs:
        print("⚠️ No documents to chunk")
        return []
    
    print(f"🔪 Chunking {len(docs)} documents...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(docs)
    
    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ============================================================================
# Embeddings and Vector Store
# ============================================================================

def get_embeddings():
    """Get embedding model"""
    print(f"🎯 Loading embeddings: {settings.EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def build_or_load_store(chunks: Optional[List[Document]] = None):
    """Build new vectorstore or load existing one"""
    embeddings = get_embeddings()
    persist_dir = str(settings.PERSIST_DIR)
    
    # Ensure directory exists
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    
    if settings.VECTOR_DB.lower() == "chroma":
        if chunks:
            # Create new vectorstore with documents
            print(f"🏗️ Building new Chroma vectorstore with {len(chunks)} chunks...")
            vs = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_dir
            )
            # Auto-persisted when persist_directory is provided!
            print("💾 Vectorstore auto-persisted to disk")
            return vs
        else:
            # Try to load existing vectorstore
            print(f"📂 Loading existing Chroma vectorstore from {persist_dir}")
            try:
                vs = Chroma(
                    embedding_function=embeddings,
                    persist_directory=persist_dir
                )
                
                # Test if vectorstore has any documents
                try:
                    test_results = vs.similarity_search("test", k=1)
                    print(f"✅ Loaded existing vectorstore with content")
                except:
                    print(f"📝 Loaded empty vectorstore")
                
                return vs
                
            except Exception as e:
                print(f"⚠️ No existing vectorstore found: {e}")
                # Create empty vectorstore
                vs = Chroma(
                    embedding_function=embeddings,
                    persist_directory=persist_dir
                )
                return vs
    
    else:
        raise ValueError(f"Unsupported vector database: {settings.VECTOR_DB}")

def get_vectorstore_stats(vectorstore):
    """Get statistics about the vectorstore"""
    if vectorstore is None:
        return {
            "status": "not_initialized",
            "type": "none"
        }
    
    try:
        # Try to get some basic stats
        test_search = vectorstore.similarity_search("test", k=5)
        return {
            "status": "loaded",
            "sample_documents": len(test_search),
            "type": "chroma"
        }
    except Exception as e:
        return {
            "status": "empty or error",
            "error": str(e),
            "type": "chroma"
        }
