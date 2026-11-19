"""
The offline phase. 
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document
import re
import unicodedata

pdfs = [
    ("data/nist_ai_100_1.pdf", "NIST AI RMF 1.0"),
    ("data/NIST_AI_600_1.pdf", "NIST AI RMF: Generative AI Profile"),
]

pages: list[Document] = []

for path, doc_title in pdfs:
    reader = PdfReader(path)
    for page_num, page in enumerate(reader.pages, start=1):  # 1-indexed for citations
        text = (page.extract_text() or "").strip()  
        if text:
            pages.append(
                Document(
                    page_content=text,
                    metadata={"source": doc_title, "page": page_num},
                )
            )

#print(pages[89])
#print(len(pages))
#on printing, we see a few non-ascii chars, eg, Eﬀectiveness and diﬃcult


def clean_pdf_text(text: str) -> str:
    #Expand all Unicode ligatures (ﬀ -> ff, ﬁ -> fi, etc.)
    text = unicodedata.normalize("NFKC", text)
    #Rejoin words broken by end-of-line hyphens (e.g. "diﬃ-\ncult" -> "difficult")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    return text.strip()

for doc in pages:
    doc.page_content = clean_pdf_text(doc.page_content)

#print(pages[89])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(pages)

# print(type(chunks))
# print(chunks[14:17])

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv() 

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    output_dimensionality=768,
)

vector_store = Chroma(
    collection_name="chunks",
    embedding_function=embeddings,
    persist_directory="vectorstore",
)

#vector_store.add_documents(chunks) - Hits free Requests per minute ceiling
import time

for i in range(0, len(chunks), 50):
    vector_store.add_documents(chunks[i : i + 50])
    print(f"Embedded {min(i + 50, len(chunks))}/{len(chunks)}")
    time.sleep(35)

print("Total chunks saved:", vector_store._collection.count())