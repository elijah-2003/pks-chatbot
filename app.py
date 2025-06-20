import streamlit as st
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# --- SETTINGS ---
FOLDER_ID = "17nNmE7bRgOgISi5MRdKOB3k1wo7ZRijE"

# --- Load FAISS index from Google Drive ---
@st.cache_resource
def load_index_from_drive(folder_id):
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    # Query the only file in the folder
    results = service.files().list(
        q=f"'{folder_id}' in parents",
        fields="files(id, name)",
        pageSize=1
    ).execute()
    files = results.get("files", [])
    if not files:
        raise RuntimeError("No files found in folder!")
    file_id = files[0]['id']
    file_name = files[0]['name']

    # Download it
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)

    # Unpickle FAISS data
    data = pickle.load(fh)
    return data, file_name

# --- Load embedding model ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --- UI ---
st.title("📘 PKS Q&A Bot")
st.caption("Ask me anything about the house.")

# Load index and model
with st.spinner("Loading index..."):
    data, filename = load_index_from_drive(FOLDER_ID)
    st.success(f"Loaded index from: {filename}")
index = data["index"]
chunks = data["chunks"]
model = load_model()

# Query box
query = st.text_input("🔍 Your question:")

if query:
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), k=3)
    st.markdown("### 🔎 Relevant Info:")
    for i in I[0]:
        st.markdown(f"- {chunks[i]['text']}")
