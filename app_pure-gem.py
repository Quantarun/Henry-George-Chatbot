__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# app.py
import os
import streamlit as st
import pinecone
from groq import Groq
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import PyPDF2
import re
import uuid
import time
from typing import List, Dict, Tuple
import random

# Configure the page
st.set_page_config(
    page_title="Henry George Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
# Handle API keys with fallback for development
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
    PINECONE_API_KEY = st.secrets.get("PINECONE_API_KEY", "")
except:
    # Fallback for development
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
    
    if not GEMINI_API_KEY:
        st.error("No Gemini API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")
    if not GROQ_API_KEY:
        st.error("No Groq API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")
    if not PINECONE_API_KEY:
        st.error("No Pinecone API key found. Please set it up in .streamlit/secrets.toml or as an environment variable.")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
EMBEDDING_DIMENSION = 1536  # Dimension for Gemini embedding
EMBEDDING_MODEL = "gemini-embedding-exp-03-07"
PINECONE_INDEX_NAME = "hg-index"
PINECONE_CLOUD = "aws"  # or "gcp"
PINECONE_REGION = "us-east-1"  # Choose appropriate region
LLM_MODEL = "gemini/gemini-2.0-flash"

# Advanced configuration
SEARCH_CONFIG = {
    "default_results": 5,
    "max_results": 10,
    "min_similarity": 0.65  # Minimum similarity score for returned documents
}

# Initialize API clients
if GEMINI_API_KEY:
    from litellm import completion
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
    # Set up Gemini API key
    genai.configure(api_key=GEMINI_API_KEY)
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    groq_client = Groq(api_key=GROQ_API_KEY)
if PINECONE_API_KEY:
    pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)

class PDFProcessor:
    """Handles all PDF processing operations with metadata extraction"""
    
    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text content and metadata from a PDF file"""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pages_data = []
        
        # Extract text and metadata from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            # Get page metadata if available
            page_metadata = {
                "page_number": page_num + 1,
                "page_size": (page.mediabox.width, page.mediabox.height)
            }
            
            pages_data.append({
                "text": text,
                "metadata": page_metadata
            })
            
        return pages_data
    
    def detect_structure(self, pages_data: List[Dict]) -> Dict:
        """Detect document structure like chapters and sections"""
        structure = {
            "chapters": [],
            "sections": []
        }
        
        # Common patterns for chapter headings
        chapter_patterns = [
            r"(?:CHAPTER|Chapter)\s+(\d+|[IVXLCDM]+)(?:\s*:\s*)?([^\n]+)?",
            r"(?:\d+\.\s+)([A-Z][^\.]+)(?:\n|\.$)",
            r"^([A-Z][A-Z\s]+)$"  # All caps headings
        ]
        
        # Simplified chapter detection
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            text = page_data["text"]
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Check for chapter headings
                for pattern in chapter_patterns:
                    match = re.search(pattern, line)
                    if match:
                        if match.groups() and len(match.groups()) > 1:
                            chapter_num = match.group(1)
                            chapter_title = match.group(2) if len(match.groups()) > 1 else ""
                        else:
                            chapter_num = len(structure["chapters"]) + 1
                            chapter_title = match.group(1) if match.groups() else line
                            
                        chapter = {
                            "number": chapter_num,
                            "title": chapter_title.strip() if chapter_title else line,
                            "start_page": page_idx + 1,
                            "start_line": line_idx
                        }
                        structure["chapters"].append(chapter)
                        break
        
        # Set end pages for chapters
        for i in range(len(structure["chapters"])):
            if i < len(structure["chapters"]) - 1:
                structure["chapters"][i]["end_page"] = structure["chapters"][i+1]["start_page"] - 1
            else:
                structure["chapters"][i]["end_page"] = len(pages_data)
        
        return structure
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\d+\n', '\n', text)
        # Remove common PDF artifacts
        text = re.sub(r'(\(cid:\d+\))', '', text)
        return text.strip()
    
    def chunk_text(self, pages_data: List[Dict], structure: Dict) -> List[Dict]:
        """Split text into chunks with metadata"""
        chunks = []
        
        # Helper function to find chapter for a given page
        def find_chapter(page_num):
            for chapter in structure["chapters"]:
                if chapter["start_page"] <= page_num <= chapter["end_page"]:
                    return {
                        "number": chapter["number"],
                        "title": chapter["title"]
                    }
            return None
        
        # Process each page and create chunks
        current_chunk_text = []
        current_chunk_size = 0
        current_page_start = 0
        current_chapter = None
        
        for page_idx, page_data in enumerate(pages_data):
            page_num = page_idx + 1
            page_text = self.clean_text(page_data["text"])
            page_lines = page_text.split('\n')
            
            # Get chapter for this page
            chapter = find_chapter(page_num)
            
            # Process page line by line
            for line in page_lines:
                words = line.split()
                
                # Add words to current chunk
                for word in words:
                    current_chunk_text.append(word)
                    current_chunk_size += len(word) + 1  # +1 for space
                
                # If chunk is large enough and at a good breaking point
                if current_chunk_size >= CHUNK_SIZE and line.endswith(('.', '?', '!')):
                    chunk_text = ' '.join(current_chunk_text)
                    chunk_id = str(uuid.uuid4())
                    
                    metadata = {
                        "chunk_id": chunk_id,
                        "start_page": current_page_start,
                        "end_page": page_num,
                    }
                    
                    # Add chapter info if available
                    if current_chapter:
                        metadata["chapter_number"] = current_chapter["number"]
                        metadata["chapter_title"] = current_chapter["title"]
                    
                    chunks.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "metadata": metadata
                    })
                    
                    # Keep some overlap for context
                    overlap_words = min(len(current_chunk_text), CHUNK_OVERLAP // 10)
                    current_chunk_text = current_chunk_text[-overlap_words:]
                    current_chunk_size = sum(len(word) + 1 for word in current_chunk_text)
                    current_page_start = page_num
                    current_chapter = chapter
        
        # Add the final chunk if there's anything left
        if current_chunk_text:
            chunk_text = ' '.join(current_chunk_text)
            chunk_id = str(uuid.uuid4())
            
            metadata = {
                "chunk_id": chunk_id,
                "start_page": current_page_start,
                "end_page": len(pages_data),
            }
            
            # Add chapter info if available
            if current_chapter:
                metadata["chapter_number"] = current_chapter["number"]
                metadata["chapter_title"] = current_chapter["title"]
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": metadata
            })
            
        return chunks

class VectorDB:
    """Handles vector database operations for the RAG system"""
    
    def __init__(self):
        """Initialize the vector database client"""
        self.pc = pinecone_client
        
        # Check if the index exists, and create it if it doesn't
        self._create_index_if_not_exists()
        
        # Connect to the index
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
    
    def _create_index_if_not_exists(self):
        """Create the index if it doesn't already exist"""
        # Check if the index exists
        if PINECONE_INDEX_NAME not in self.pc.list_indexes().names():
            # Create the index
            self.pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=pinecone.ServerlessSpec(
                    cloud=PINECONE_CLOUD,
                    region=PINECONE_REGION
                )
            )
            
            # Wait for the index to be ready
            while not self.pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
    
    def _create_embedding(self, text: str) -> list:
        """Create an embedding vector for the given text using Gemini"""
        try:
            # Check if text is too long for Gemini's token limit
            if len(text) > 25000:  # Approximation, adjust as needed
                # If text is too long, truncate it to fit within token limits
                text = text[:25000]  # Simple truncation
            
            # Create embedding using Gemini
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            
            # Return the embedding as a list
            embedding = result["embedding"]
            
            # Check if all values are zero (which Pinecone rejects)
            if all(x == 0 for x in embedding):
                # Add a tiny random non-zero value to avoid Pinecone's rejection
                import random
                index_to_modify = random.randint(0, len(embedding) - 1)
                embedding[index_to_modify] = 0.0001
                print(f"Warning: Zero vector detected and fixed for text: {text[:50]}...")
                
            return embedding
        except Exception as e:
            print(f"Error creating embedding: {str(e)}")
            # Create a fallback embedding that isn't all zeros
            fallback = [0.0] * EMBEDDING_DIMENSION
            fallback[0] = 0.0001  # Add a small non-zero value
            return fallback
    
    def add_documents(self, chunks: list, namespace: str):
        """Add document chunks to the vector database"""
        # Process chunks in batches
        batch_size = 50  # Reduced batch size for safety
        total_chunks = len(chunks)
        
        for i in range(0, total_chunks, batch_size):
            # Create a batch of chunks
            batch = chunks[i:min(i+batch_size, total_chunks)]
            
            # Create vectors for batch upsert
            vectors = []
            for chunk in batch:
                try:
                    # Create embedding for the text
                    embedding = self._create_embedding(chunk["text"])
                    
                    # Limit metadata size - only include essential fields
                    # Pinecone has a 40KB metadata size limit per vector
                    truncated_text = chunk["text"]
                    if len(truncated_text) > 8000:  # Roughly 20KB of text
                        truncated_text = truncated_text[:8000] + "..."
                    
                    # Create streamlined metadata
                    metadata = {
                        "text": truncated_text,
                        "start_page": chunk["metadata"].get("start_page", 0),
                        "end_page": chunk["metadata"].get("end_page", 0)
                    }
                    
                    # Only add chapter info if present
                    if "chapter_number" in chunk["metadata"]:
                        metadata["chapter_number"] = chunk["metadata"]["chapter_number"]
                    if "chapter_title" in chunk["metadata"]:
                        chapter_title = chunk["metadata"]["chapter_title"]
                        # Truncate long chapter titles
                        if len(chapter_title) > 200:
                            chapter_title = chapter_title[:200] + "..."
                        metadata["chapter_title"] = chapter_title
                    
                    # Create a vector record
                    vectors.append({
                        "id": chunk["id"],
                        "values": embedding,
                        "metadata": metadata
                    })
                except Exception as e:
                    print(f"Error processing chunk {chunk['id']}: {str(e)}")
                    # Continue with the next chunk
                    continue
            
            # Skip empty batches
            if not vectors:
                continue
                
            try:
                # Upsert vectors to Pinecone
                self.index.upsert(vectors=vectors, namespace=namespace)
                print(f"Successfully upserted batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}")
            except Exception as e:
                print(f"Error upserting batch to Pinecone: {str(e)}")
                # Continue with the next batch
                continue
    
    def search(self, 
               query: str, 
               namespace: str,
               n_results: int = 5) -> list:
        """Search for relevant documents"""
        # Create query embedding
        query_embedding = self._create_embedding(query)
        
        # Perform the search
        search_params = {
            "namespace": namespace,
            "vector": query_embedding,
            "top_k": n_results,
            "include_metadata": True
        }
        
        search_results = self.index.query(**search_params)
        
        # Format results
        results = []
        for match in search_results.matches:
            results.append({
                "id": match.id,
                "text": match.metadata.get("text", ""),
                "metadata": match.metadata,
                "score": match.score
            })
            
        return results
    
    def get_chapter_info(self, namespace: str) -> list:
        """Get chapter information from the database"""
        try:
            # Get a sample of vectors to extract chapter metadata
            sample_results = self.index.query(
                namespace=namespace,
                vector=[0.0001] * EMBEDDING_DIMENSION,  # Non-zero dummy vector
                top_k=100,
                include_metadata=True
            )
            
            # Extract unique chapter information
            chapter_info = {}
            for match in sample_results.matches:
                metadata = match.metadata
                if "chapter_number" in metadata and "chapter_title" in metadata:
                    chapter_key = metadata["chapter_number"]
                    if chapter_key not in chapter_info:
                        chapter_info[chapter_key] = {
                            "number": metadata["chapter_number"],
                            "title": metadata["chapter_title"]
                        }
            
            # Convert to list and sort by chapter number
            chapters_list = list(chapter_info.values())
            return chapters_list
        except Exception as e:
            print(f"Error fetching chapter info: {str(e)}")
            return []
            
    def list_namespaces(self) -> List[str]:
        """List all namespaces in the index"""
        try:
            stats = self.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys())
            return namespaces
        except Exception as e:
            print(f"Error listing namespaces: {str(e)}")
            return []

class RAGChatbot:
    """Main chatbot class that uses RAG with Groq"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.vector_db = VectorDB()
        self.pdf_processor = PDFProcessor()
        self.conversation_history = []
        
    def process_pdf(self, pdf_file, namespace: str):
        """Process a PDF and store in vector database"""
        # Extract text and basic metadata
        pages_data = self.pdf_processor.extract_text_from_pdf(pdf_file)
        
        # Detect document structure (chapters, sections)
        structure = self.pdf_processor.detect_structure(pages_data)
        
        # Chunk the text with structural metadata
        chunks = self.pdf_processor.chunk_text(pages_data, structure)
        
        # Store in vector DB
        self.vector_db.add_documents(chunks, namespace)
        
        # Generate summary statistics
        total_text = " ".join([page["text"] for page in pages_data])
        chapter_info = [
            {
                "number": chapter.get("number", "N/A"),
                "title": chapter.get("title", "Untitled"),
                "pages": f"{chapter.get('start_page', 'N/A')}-{chapter.get('end_page', 'N/A')}"
            }
            for chapter in structure["chapters"]
        ]
        
        return {
            "namespace": namespace,
            "chunks_count": len(chunks),
            "total_chars": len(total_text),
            "chapter_count": len(structure["chapters"]),
            "chapter_info": chapter_info[:5] if chapter_info else []  # Show first 5 chapters
        }
    
    def _classify_query(self, query: str) -> str:
        """Classify the query type"""
        # Default to book question
        return "book_question"
    
    def query(self, namespaces: List[str], user_query: str) -> Tuple[str, List[Dict]]:
        """Query the RAG system across multiple namespaces and generate a response"""
        # Classify the query type
        query_type = self._classify_query(user_query)
        
        all_search_results = []
        
        # Search each namespace and collect results
        for namespace in namespaces:
            namespace_results = self.vector_db.search(
                query=user_query,
                namespace=namespace,
                n_results=max(1, SEARCH_CONFIG["default_results"] // len(namespaces))  # Distribute results fairly
            )
            
            # Add source namespace to each result for citation
            for result in namespace_results:
                result["namespace"] = namespace
                # Add a display name for the namespace
                result["book_name"] = namespace.replace('book_', '').replace('_', ' ').title()
            
            all_search_results.extend(namespace_results)
        
        # Sort results by relevance score
        all_search_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to top results
        search_results = all_search_results[:SEARCH_CONFIG["default_results"]]
        
        # If no results found, handle appropriately
        if not search_results:
            no_info_response = "I couldn't find relevant information to answer your question. Could you try rephrasing or asking something else?"
            self.conversation_history.append({
                "query": user_query,
                "response": no_info_response,
                "type": "no_results"
            })
            return no_info_response, []
        
        # Prepare context from search results
        context_parts = []
        for i, result in enumerate(search_results):
            # Build context entry
            context_entry = f"Document {i+1} (from {result['book_name']}):"
            metadata = result["metadata"]
            
            # Add chapter information if available
            if "chapter_title" in metadata:
                context_entry += f" Chapter: {metadata.get('chapter_number', '')}: {metadata.get('chapter_title', '')}"
                
            # Add page information
            context_entry += f" (Pages {metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')})"
            
            # Add the document text
            context_entry += f"\n{result['text']}"
            
            context_parts.append(context_entry)
            
        context = "\n\n".join(context_parts)
        
        # Create prompt for LLM
        prompt = f"""
Persona:
You are Henry George (1839-1897), the American political economist, journalist, social reformer, and influential orator. You are speaking with the passion, clarity, and moral conviction characteristic of your major works like Progress and Poverty and Protection or Free Trade, as well as your public speeches. You engage energetically with the great social and economic questions of your time, aiming to diagnose the root causes of injustice and persuade others of the necessary remedies.
Core Beliefs & Knowledge (Your Unwavering Ideology):
Central Thesis (Georgism): Your core belief is that while individuals should own the value they create through their labor and capital, the economic value derived from land (including all natural resources and opportunities) rightly belongs to the community as a whole, as its value arises from the presence and development of society, not individual effort.
The Land Question is Paramount: You see the private monopolization of land and the private capture of its economic rent as the fundamental cause of the paradox of "progress and poverty" – the persistence and worsening of poverty and inequality alongside technological and economic advancement. This system denies labor access to natural opportunities and is equivalent to a form of slavery.
Land Value Taxation (The "Single Tax"): You advocate vigorously for abolishing taxes on productive activities (wages, capital, improvements, trade) and deriving public revenue primarily, or solely, by taxing the value of land itself. This is not a tax on land area, but on its market value conferred by the community.
Benefits: This tax cannot be shifted to tenants or consumers, encourages the best use of land, discourages harmful speculation, simplifies government, raises wages, increases opportunities, and captures community-created value for public benefit.
Distinction: Clarify that this is not land nationalization or confiscation of land titles. Individuals can retain possession, buy, sell, and bequeath land; society simply collects the annual value (rent) they did not create. ("We may safely leave them the shell, if we take the kernel.")
Free Trade: You are an ardent opponent of protectionist tariffs. You argue they artificially raise prices for consumers, harm overall economic activity, protect inefficient monopolies, fail to raise general wages, and corrupt politics. You see free trade as aligned with natural economic laws and international cooperation.
Natural Monopolies: You believe services that inherently require exclusive rights-of-way or control over essential networks (e.g., railroads, utilities like water/electricity, telegraphs, potentially urban transit) are "natural monopolies." These should ideally be under public ownership or strict regulation (municipalization) and often provided at cost or even free, funded by the land value increases they generate.
Money and Finance: You support "debt-free" sovereign money (like greenbacks) issued by the government, capturing seigniorage for public benefit. You are critical of metallic currency limitations and fiat money created by private banks, viewing much credit/debt as tied to land speculation rather than productive enterprise. You advocate for bankruptcy protections and oppose debtors' prisons, seeing much debt as illegitimate claims on rent.
Political & Social Reforms: You are a strong advocate for:
Secret Ballot: (Crucial for preventing bribery and ensuring voter freedom).
Women's Suffrage: (A matter of fundamental political rights).
Citizen's Dividend/Universal Pension: Surplus land rent revenues could fund basic incomes or pensions distributed "as a right," not charity.
Intellectual Property Skepticism: View patents and copyrights cautiously, as forms of monopoly potentially extracting unearned rent, though perhaps less harmful than land monopoly.
Reduced Military Spending & Civil Service Reform.
Tone and Style:
Passionate and Earnest: Speak with profound conviction about justice, rights, and the moral imperative of your reforms.
Reasoned and Didactic: Explain complex economic ideas with clarity, logic, and accessible language. Use analogies and examples (both historical and hypothetical) effectively.
Moralistic and Righteous: Frame arguments in terms of natural law, the Creator's intent (using a deistic/humanitarian lens), justice vs. injustice, equality, and the common good.
Confident and Assertive: Directly address and dismantle opposing arguments (like those for protectionism or the current tax system) with robust logic.
Populist: Connect with the concerns of ordinary people, laborers, and producers, contrasting their struggles with the unearned income of monopolists.
Engaging and Oratorical: Employ rhetorical questions, address the user respectfully (e.g., "friend," "fellow-citizen"), and maintain a somewhat formal but powerful speaking style appropriate to the late 19th century.
Historically Grounded: Speak from the perspective of your era, referencing contemporary events, figures (like Hewitt, Powderly, Marx – often critically), and intellectual currents. Avoid anachronisms.
Goal:
Your purpose is to educate, inspire, and persuade the user of the fundamental truth and justice of your core ideas – particularly the necessity of Land Value Taxation as the foundation for a just and prosperous society. You aim to show how this central reform connects to and supports other vital reforms (like free trade and public control of monopolies), ultimately leading to the abolition of involuntary poverty, the fair distribution of wealth, and the realization of true liberty and equality for all members of the community. You seek to clarify your positions and counter misrepresentations.
Give Short responses and to the point(around 100 words). Unless asked for a detailed response.
You should use the 19th century American English for your responses.

        Context from the books:
        {context}
        
        Question: {user_query}
        
        Answer:
        """
        
        # Generate response using Groq
        response = completion(
            model=LLM_MODEL,  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides well-cited answers about books."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        response_text = response.choices[0].message.content
        
        # Create citation data for displaying sources
        citations = []
        for i, result in enumerate(search_results):
            metadata = result["metadata"]
            citation = {
                "number": i + 1,
                "id": result["id"],
                "text": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"],
                "book_name": result["book_name"],
                "namespace": result["namespace"],
                "metadata": {}
            }
            
            # Add structural information to citations if available
            if "chapter_title" in metadata:
                citation["metadata"]["chapter"] = f"{metadata.get('chapter_number', '')}: {metadata.get('chapter_title', '')}"
                
            citation["metadata"]["pages"] = f"{metadata.get('start_page', 'N/A')}-{metadata.get('end_page', 'N/A')}"
            
            # Add relevance score
            similarity_score = round(result["score"] * 100, 1)
            citation["metadata"]["relevance"] = f"{similarity_score}%"
            
            citations.append(citation)
        
        # Save to conversation history
        self.conversation_history.append({
            "query": user_query,
            "response": response_text,
            "type": query_type,
            "citations": [c["id"] for c in citations]
        })
        
        return response_text, citations

# Initialize the chatbot
chatbot = RAGChatbot()

# Simple UI
st.title("📚 Henry George Chatbot")

# Sidebar for document list and file upload
with st.sidebar:
    st.header("Available Documents")
    
    # Get list of available namespaces (books)
    available_namespaces = chatbot.vector_db.list_namespaces()
    
    # Display all available books
    if not available_namespaces:
        st.info("No documents found in the database. You can upload a book below.")
    else:
        # Format namespace names for display
        display_names = [ns.replace('book_', '').replace('_', ' ').title() for ns in available_namespaces]
        
        st.write(f"**{len(available_namespaces)} documents available:**")
        for book_name in display_names:
            st.write(f"- {book_name}")
        
        # Store all namespaces in session state for querying
        st.session_state.all_namespaces = available_namespaces
    
    # File upload section
    st.divider()
    st.header("Upload Document (Optional)")
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")
    
    if uploaded_file:
        # Create a valid namespace name
        default_name = re.sub(r'[^a-zA-Z0-9._-]', '_', uploaded_file.name.replace('.pdf', ''))
        if not default_name[0].isalnum():
            default_name = 'b_' + default_name
        if len(default_name) > 63:
            default_name = default_name[:60] + '_db'
            
        namespace = f"book_{default_name}"
        
        if st.button("Process Document"):
            with st.spinner("Processing your document... This may take a while for large files."):
                # Process the book
                processing_result = chatbot.process_pdf(uploaded_file, namespace)
                st.success("Document processed successfully!")
                
                # Display simple stats
                st.write(f"Found {processing_result['chapter_count']} chapters")
                st.write(f"Created {processing_result['chunks_count']} text chunks")
                
                # Refresh the page to update available documents
                st.rerun()
    
    # Document management
    if available_namespaces:
        st.divider()
        with st.expander("Document Management"):
            # Select a document to delete
            book_to_delete = st.selectbox(
                "Select document to delete:",
                options=display_names,
                index=None
            )
            
            if book_to_delete and st.button("Delete Selected Document"):
                try:
                    # Convert display name back to namespace
                    namespace_to_delete = available_namespaces[display_names.index(book_to_delete)]
                    
                    # Delete namespace instead of entire index
                    chatbot.vector_db.index.delete(
                        namespace=namespace_to_delete,
                        delete_all=True
                    )
                    st.success(f"Document '{book_to_delete}' deleted!")
                    # Refresh the page
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # About section
    st.divider()
    st.write("Made with Pinecone, Gemini and Groq")

# Chat tab (default view)
#with tab1:
    # Sidebar for book information 
    with st.sidebar:
        st.header("Available Books")
        
        # Get list of available namespaces (books)
        available_namespaces = chatbot.vector_db.list_namespaces()
        
        if not available_namespaces:
            st.info("No books found in the database. Please upload a book first.")
        else:
            # Format namespace names for display
            display_names = [ns.replace('book_', '').replace('_', ' ').title() for ns in available_namespaces]
            
            # Display all available books
            st.write(f"**{len(available_namespaces)} books available for chat:**")
            for book_name in display_names:
                st.write(f"- {book_name}")
            
            # Option to manage books
            with st.expander("Book Management"):
                # Select a book to delete
                book_to_delete = st.selectbox(
                    "Select book to delete:",
                    options=display_names,
                    index=None
                )
                
                if book_to_delete and st.button("Delete Selected Book"):
                    try:
                        # Convert display name back to namespace
                        namespace_to_delete = available_namespaces[display_names.index(book_to_delete)]
                        
                        # Delete namespace instead of entire index
                        chatbot.vector_db.index.delete(
                            namespace=namespace_to_delete,
                            delete_all=True
                        )
                        st.success(f"Book '{book_to_delete}' data deleted!")
                        # Refresh the page
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            # Store all namespaces in session state for querying
            st.session_state.all_namespaces = available_namespaces
        
        # About section
        st.divider()
        st.write("Made with Pinecone, Gemini and Groq")

    # Initialize chat history if not already done
    if "messages" not in st.session_state:
        st.session_state.messages = []

# Main chat area
st.subheader("Chat with Henry George")

# Display chat history
for message_idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If the message has citations, show them in an expander
        if message["role"] == "assistant" and "citations" in message and message["citations"]:
            with st.expander("View Sources"):
                for citation_idx, citation in enumerate(message["citations"]):
                    st.markdown(f"**Source {citation['number']}**")
                    
                    # Display metadata if available
                    if "metadata" in citation:
                        metadata = citation["metadata"]
                        cols = st.columns(2)
                        
                        # Show chapter information
                        if "chapter" in metadata:
                            cols[0].write(f"Chapter: {metadata['chapter']}")
                        
                        # Show page range
                        if "pages" in metadata:
                            cols[1].write(f"Pages: {metadata['pages']}")
                    
                    # Display excerpt with unique key using both message and citation indices
                    st.text_area("Excerpt", citation["text"], height=100, 
                                key=f"history_source_{message_idx}_{citation_idx}")
                    st.divider()

# Only show chat input if there are namespaces available
if "all_namespaces" in st.session_state and st.session_state.all_namespaces:
    user_query = st.chat_input("Ask a question about any of the books")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching books and generating response..."):
                try:
                    response_text, citations = chatbot.query(
                        st.session_state.all_namespaces,
                        user_query
                    )
                    
                    st.markdown(response_text)
                    
                    # Show citations if they exist
                    if citations:
                        with st.expander("View Sources"):
                            for citation_idx, citation in enumerate(citations):
                                # Include book name in the source header
                                st.markdown(f"**Source {citation['number']} - From {citation['book_name']}**")
                                
                                # Display metadata
                                if "metadata" in citation:
                                    metadata = citation["metadata"]
                                    cols = st.columns(2)
                                    
                                    # Show chapter information
                                    if "chapter" in metadata:
                                        cols[0].write(f"Chapter: {metadata['chapter']}")
                                    
                                    # Show page range
                                    if "pages" in metadata:
                                        cols[1].write(f"Pages: {metadata['pages']}")
                                
                                # Display excerpt with unique key using the citation index
                                st.text_area("Excerpt", citation["text"], height=100, key=f"current_source_{citation_idx}")
                                st.divider()
                    
                    # Save to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response_text,
                        "citations": citations
                    })
                    
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"I'm sorry, I encountered an error. Please try again.",
                        "citations": []
                    })
else:
    # If no books are available yet, show instructions
    st.info("No books found in the database. Please upload a book in the 'Upload New Book' tab to get started.")
