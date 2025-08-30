"""
vectorize_sources.py

プロジェクトルート下のデータソースを読み込み、Embed + VectorStore(FAISS)化して
"project_root/vectorized/<mode>/" に保存するユーティリティ。

対応フォーマット (試みるもの): .txt, .md, .csv, .pdf, .docx, .pptx
(Unstructured や PyPDFLoader 等が無ければテキスト読み込みにフォールバックします)

使い方例:
    # OpenAI embeddings を使う (OPENAI_API_KEY を .env に記載)
    python vectorize_sources.py \
        --chat_dir source/chat_mode \
        --code_dir source/code_mode \
        --out_dir vectorized \
        --emb_backend openai

    # Azure OpenAI embeddings を使う（Azure の設定を .env に記載）
    python vectorize_sources.py --emb_backend azure

必要パッケージ(一例):
    pip install langchain faiss-cpu unstructured sentence-transformers python-docx python-pptx python-magic python-dotenv
    # OpenAI を使う場合: pip install openai

注意: 環境や langchain のバージョンにより Loader 名や VectorStore API が変わることがあります。
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# load .env
load_dotenv()

# Vector store & embeddings - 選択肢を用意
try:
    from langchain.embeddings import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

try:
    from langchain_openai import AzureOpenAIEmbeddings
except Exception:
    AzureOpenAIEmbeddings = None

try:
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    HuggingFaceEmbeddings = None

try:
    from langchain.vectorstores import FAISS
except Exception:
    FAISS = None

# いくつかの loader を try-import (なければ fallback の手作業読み込みをする)
_loader_imports = {}
try:
    from langchain.document_loaders import CSVLoader, TextLoader

    _loader_imports["text"] = TextLoader
    _loader_imports["csv"] = CSVLoader
except Exception:
    pass

try:
    # PyPDFLoader / UnstructuredPDFLoader を試す
    from langchain.document_loaders import PyPDFLoader

    _loader_imports["pdf"] = PyPDFLoader
except Exception:
    try:
        from langchain.document_loaders import UnstructuredPDFLoader

        _loader_imports["pdf"] = UnstructuredPDFLoader
    except Exception:
        pass

try:
    from langchain.document_loaders import Docx2txtLoader

    _loader_imports["docx"] = Docx2txtLoader
except Exception:
    pass

try:
    from langchain.document_loaders import UnstructuredWordDocumentLoader

    _loader_imports["docx"] = UnstructuredWordDocumentLoader
except Exception:
    pass

try:
    from langchain.document_loaders import UnstructuredPowerPointLoader

    _loader_imports["pptx"] = UnstructuredPowerPointLoader
except Exception:
    pass


def find_files(root: Path, exts: List[str]) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def load_file_to_docs(path: Path) -> List[Document]:
    """拡張子に応じて最適な loader を使って Document のリストを返す。
    loader がなければ単純にテキストを読み取って 1 ドキュメントにする。
    """
    ext = path.suffix.lower()
    try:
        if ext in (".txt", ".md") and "text" in _loader_imports:
            loader = _loader_imports["text"](str(path), encoding="utf-8")
            return loader.load()
        if ext == ".csv" and "csv" in _loader_imports:
            loader = _loader_imports["csv"](str(path))
            return loader.load()
        if ext == ".pdf" and "pdf" in _loader_imports:
            loader = _loader_imports["pdf"](str(path))
            return loader.load()
        if ext == ".docx" and "docx" in _loader_imports:
            loader = _loader_imports["docx"](str(path))
            return loader.load()
        if ext == ".pptx" and "pptx" in _loader_imports:
            loader = _loader_imports["pptx"](str(path))
            return loader.load()
    except Exception as e:
        print(f"[warn] loader failed for {path}: {e}")

    # fallback: raw text read
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            # 二進読み込みして decode を試す
            text = path.open("rb").read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    if not text:
        return []
    meta = {"source": str(path)}
    return [Document(page_content=text, metadata=meta)]


def docs_from_dir(root: Path) -> List[Document]:
    if not root.exists():
        return []
    exts = [".txt", ".md", ".csv", ".pdf", ".docx", ".pptx"]
    files = find_files(root, exts)
    docs: List[Document] = []
    for f in files:
        ds = load_file_to_docs(f)
        if ds:
            docs.extend(ds)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return splitter.split_documents(docs)


def get_embeddings(
    backend: str = "openai",
    hf_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """emb_backend の選択肢:
    - 'openai' : 標準の OpenAI API（OPENAI_API_KEY を .env に記載）
    - 'azure'  : Azure OpenAI（AZURE_OPENAI_* を .env に記載）
    - 'hf'     : HuggingFace ローカル/リモートモデル

    注意: langchain のバージョンにより OpenAIEmbeddings のコンストラクタ引数が違う可能性があるため
    失敗した際は適宜調整してください。
    """
    backend = backend.lower()

    if backend == "openai":
        if OpenAIEmbeddings is None:
            raise RuntimeError(
                "OpenAIEmbeddings is not available. Install openai/langchain versions that include it."
            )
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment (.env)")
        # ensure env var is set for libraries that read it
        os.environ["OPENAI_API_KEY"] = openai_key
        model = os.getenv("EMBEDDING_MODEL")
        return OpenAIEmbeddings(model=model)

    elif backend == "azure":
        if AzureOpenAIEmbeddings is None:
            raise RuntimeError("AzureOpenAIEmbeddings is not available (required for Azure path).")

        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        azure_deployment = os.getenv("EMBEDDING_AZURE_DEPLOYMENT")

        if not (azure_key and azure_endpoint and azure_deployment):
            raise RuntimeError(
                "AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / EMBEDDING_AZURE_DEPLOYMENT required for azure backend"
            )

        # try to instantiate with deployment name if supported
        return AzureOpenAIEmbeddings(
            azure_endpoint=azure_endpoint,
            api_key=azure_key,
            api_version=azure_api_version,
            deployment=azure_deployment,
        )

    else:
        # HF backend
        if HuggingFaceEmbeddings is None:
            try:
                from langchain.embeddings import HuggingFaceEmbeddings as HFE

                return HFE(model_name=hf_model)
            except Exception:
                raise RuntimeError(
                    "HuggingFaceEmbeddings not available. Install sentence-transformers and langchain support."
                )
        return HuggingFaceEmbeddings(model_name=hf_model)


def build_and_save_vectorstore(docs: List[Document], embeddings, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not docs:
        print(f"[info] no documents to index for {out_dir}")
        return

    if FAISS is None:
        raise RuntimeError(
            "FAISS vectorstore is not available. Please install langchain and faiss-cpu."
        )

    print(f"[info] Splitting {len(docs)} documents...")
    docs_split = split_documents(docs)
    print(f"[info] {len(docs_split)} chunks after splitting.")

    print("[info] Building vectorstore (this may take a while)...")
    store = FAISS.from_documents(docs_split, embeddings)

    # 保存 (langchain faiss store の save_local を使う)
    try:
        store.save_local(str(out_dir))
        print(f"[info] saved vectorstore to {out_dir}")
    except Exception as e:
        # 最悪 pickle 保存
        print(f"[warn] save_local failed: {e}. falling back to pickle save")
        with open(out_dir / "faiss_store.pkl", "wb") as f:
            pickle.dump(store, f)
        print(f"[info] pickled vectorstore to {out_dir / 'faiss_store.pkl'}")


def main(chat_dir: str, code_dir: str, out_dir: str, emb_backend: str, hf_model: str):
    chat_root = Path(chat_dir)
    code_root = Path(code_dir)
    out_root = Path(out_dir)

    print(
        f"[info] chat_dir={chat_root}, code_dir={code_root}, out_dir={out_root}, emb_backend={emb_backend}"
    )

    embeddings = get_embeddings(backend=emb_backend, hf_model=hf_model)

    # チャット用データ読み込み・ベクトル化
    chat_docs = docs_from_dir(chat_root)
    print(f"[info] found {len(chat_docs)} chat documents")
    build_and_save_vectorstore(chat_docs, embeddings, out_root / "chat_mode")

    # コード用データ読み込み・ベクトル化
    code_docs = docs_from_dir(code_root)
    print(f"[info] found {len(code_docs)} code documents")
    build_and_save_vectorstore(code_docs, embeddings, out_root / "code_mode")

    print("[done]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chat_dir", type=str, default="data/source_text/chat_mode", help="chat mode source dir"
    )
    parser.add_argument(
        "--code_dir", type=str, default="data/source_text/code_mode", help="code mode source dir"
    )
    parser.add_argument(
        "--out_dir", type=str, default="vectorized", help="output dir for vector stores"
    )
    parser.add_argument(
        "--emb_backend",
        type=str,
        default="azure",
        choices=["openai", "hf", "azure"],
        help="embeddings backend (openai | azure | hf)",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="hf model name if emb_backend=hf",
    )
    args = parser.parse_args()

    main(args.chat_dir, args.code_dir, args.out_dir, args.emb_backend, args.hf_model)
