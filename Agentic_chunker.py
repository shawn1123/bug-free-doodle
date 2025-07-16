# 1. Install dependencies
!pip install -U langchain_experimental langchain-openai

# 2. Imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import openai, os

# 3. Set your OpenAI key
os.environ["OPENAI_API_KEY"] = "..."  # or set via environment

# 4. Load your text
with open("large_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

docs = [Document(page_content=text)]

# 5. Initial semantic chunker
emb = OpenAIEmbeddings()
sem_splitter = SemanticChunker(
    embeddings=emb,
    breakpoint_threshold_type="percentile",     # splits at semantic gaps
    breakpoint_threshold_amount=95.0,           # e.g., top 5% distance for breakpoints
)

initial_chunks = sem_splitter.create_documents([text])

print(f"➡️ Initial semantic chunks count: {len(initial_chunks)}")

# 6. Agentic refinement: use an LLM to decide merging or further splitting
from langchain import OpenAI
llm = OpenAI(temperature=0)

def agentic_chunking(chunks):
    refined = []
    buffer = ""
    for chunk in chunks:
        prompt = f"""
Chunk A:
\"\"\"{buffer}\"\"\"

Chunk B:
\"\"\"{chunk.page_content}\"\"\"

Should these be merged into one chunk? Answer "YES" or "NO" and justify briefly.
"""
        resp = llm(prompt).strip().upper()
        if resp.startswith("YES"):
            buffer = buffer + "\n\n" + chunk.page_content
        else:
            if buffer:
                refined.append(Document(page_content=buffer))
            buffer = chunk.page_content
    if buffer:
        refined.append(Document(page_content=buffer))
    return refined

refined_chunks = agentic_chunking(initial_chunks)
print(f"➡️ Agentically refined chunks count: {len(refined_chunks)}")

# Now `refined_chunks` contains agentically split/merged text chunks.
# You can inspect them, add metadata, or feed them into downstream systems.
