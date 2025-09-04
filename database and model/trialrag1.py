import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

# ⚠️ Use env vars in real projects
OPENAI_API_KEY = "key goes here"# -------- FILE INPUT --------
file_path = r"C:\Users\kanishk\Downloads\blogs_vector_db.parquet"


text = ""

if file_path.endswith(".pdf"):
    # PDF branch
    pdf_reader = PdfReader(file_path)
    for page in pdf_reader.pages:
        extracted = page.extract_text() or ""
        text += extracted

elif file_path.endswith((".parquet", ".parq")):
    try:
        df = pd.read_parquet(file_path)  # Needs pyarrow or fastparquet
    except Exception as e:
        raise RuntimeError(f"Failed to read Parquet. Error: {e}")

    if df.empty:
        raise ValueError("Parquet file is empty.")

    # pick text column (default: "text" if exists)
    candidate_text_cols = [c for c in df.columns if df[c].dtype == object]
    if not candidate_text_cols:
        candidate_text_cols = list(df.columns)

    sel_col = "text" if "text" in candidate_text_cols else candidate_text_cols[0]
    print(f"Using column '{sel_col}' for text...")

    max_rows = min(len(df), 2000)
    series = df[sel_col].astype(str).fillna("")
    text_list = series.head(max_rows).tolist()
    text = "\n\n".join(text_list)

    print(f"Loaded {max_rows} rows from column '{sel_col}'")

else:
    raise ValueError("Unsupported file type (only PDF/Parquet).")

# -------- CHUNKING --------
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
chunks = text_splitter.split_text(text)

# -------- EMBEDDINGS + VECTOR STORE --------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY
)
vector_store = FAISS.from_texts(chunks, embeddings)

# -------- CHAT MODEL --------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    max_tokens=1000
)
chain = load_qa_chain(llm, chain_type="stuff")

# -------- USER QUERY LOOP --------
while True:
    user_question = input("\nAsk a question (or type 'exit'): ")
    if user_question.lower() == "exit":
        break

    match = vector_store.similarity_search(user_question)
    response = chain.run(input_documents=match, question=user_question)

    print("\nAnswer:", response)
