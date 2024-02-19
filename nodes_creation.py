from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.extractors import (
    KeywordExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter

load_dotenv("/home/dai/35/AyurSanvaad/.env")

from llama_index.core import Settings, SimpleDirectoryReader

from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.gemini import Gemini

Settings.embed_model = GeminiEmbedding(model_name="models/embedding-001")
Settings.llm = Gemini(model_name="models/gemini-pro", temperature=0.5)

reader = SimpleDirectoryReader(input_dir="/home/dai/35/AyurSanvaad/data/text_data/Articles", recursive=True)
documents = reader.load_data(show_progress=True)

sentence_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
title_extractor = TitleExtractor(nodes=5, llm = Settings.llm)
qa_extractor = QuestionsAnsweredExtractor(questions=3, llm = Settings.llm)
summary_extractor = SummaryExtractor(summaries=["self"], llm = Settings.llm)
keyword_extractor = KeywordExtractor(keywords=20, llm = Settings.llm)
pipeline = IngestionPipeline(
    transformations=[sentence_splitter, keyword_extractor]
)

nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)

#save nodes in pickle file
import pickle
with open("/home/dai/35/AyurSanvaad/strorage/Article_nodes.pkl", "wb") as f:
    pickle.dump(nodes, f)

def save_docstore(nodes, save_dir: str, store_name: str) -> None:
    """
    Create a document store and save it to the specified directory.

    Args:
        nodes (List[str]): List of nodes to be added to the document store.
        save_dir (str): Directory path where the document store will be saved.
        store_name (str): Name of the document store file.

    Returns:
        None
    """
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    docstore.add_documents(nodes)
    docstore.persist(persist_path=save_dir/store_name)
    return docstore
try:
    save_docstore(nodes, "/home/dai/35/AyurSanvaad/strorage/docstore", "Articles")
except:
    print("Error in saving docstore")