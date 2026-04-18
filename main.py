"""
Ollama RAG - 폴더 내 문서를 기억하고 질문에 답하는 프로그램
지원 파일: .txt, .docx
"""

import os
import sys
import argparse

# Windows 터미널 한글 출력 설정
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import hashlib
from pathlib import Path

try:
    import ollama
    import chromadb
    from docx import Document as DocxDocument
except ImportError as e:
    print(f"[오류] 필요한 패키지가 없습니다: {e}")
    print("setup.bat 를 먼저 실행하세요.")
    sys.exit(1)


# 청크 설정
CHUNK_SIZE = 500        # 청크당 최대 글자 수
CHUNK_OVERLAP = 100     # 청크 간 겹치는 글자 수
EMBED_MODEL = "nomic-embed-text"   # 임베딩 모델
COLLECTION_NAME = "documents"


def read_txt(path: Path) -> str:
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def read_docx(path: Path) -> str:
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def load_documents(folder: Path) -> list[dict]:
    docs = []
    patterns = [("*.txt", read_txt), ("*.docx", read_docx)]
    for pattern, reader in patterns:
        for file in folder.rglob(pattern):
            try:
                text = reader(file)
                if text.strip():
                    docs.append({"path": str(file), "text": text})
                    print(f"  읽음: {file.name} ({len(text):,}자)")
            except Exception as e:
                print(f"  [건너뜀] {file.name}: {e}")
    return docs


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def file_hash(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def get_chroma_client(db_path: str):
    return chromadb.PersistentClient(path=db_path)


def index_documents(folder: Path, db_path: str, force: bool = False):
    print(f"\n폴더 스캔 중: {folder}")
    docs = load_documents(folder)
    if not docs:
        print("지원하는 파일(.txt, .docx)이 없습니다.")
        return

    client = get_chroma_client(db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 이미 인덱싱된 파일 해시 추적
    existing_meta = {}
    try:
        results = collection.get(include=["metadatas"])
        for meta in results["metadatas"]:
            if meta and "file_hash" in meta:
                existing_meta[meta["file_hash"]] = True
    except Exception:
        pass

    added = 0
    skipped = 0
    for doc in docs:
        fhash = file_hash(doc["path"])
        if not force and fhash in existing_meta:
            skipped += 1
            continue

        # 같은 파일의 기존 청크 삭제 후 재추가
        try:
            existing = collection.get(where={"source": doc["path"]})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
        except Exception:
            pass

        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                resp = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
                embedding = resp["embedding"]
                chunk_id = f"{fhash}_{i}"
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc["path"],
                        "file_hash": fhash,
                        "chunk_index": i,
                        "filename": Path(doc["path"]).name,
                    }]
                )
                added += 1
            except Exception as e:
                print(f"  [오류] 임베딩 실패 ({Path(doc['path']).name} 청크 {i}): {e}")

    print(f"\n인덱싱 완료: {added}개 청크 추가, {skipped}개 파일 건너뜀(변경 없음)")


def query(question: str, db_path: str, llm_model: str, top_k: int = 5) -> str:
    client = get_chroma_client(db_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return "문서가 아직 인덱싱되지 않았습니다. 먼저 'index' 명령을 실행하세요."

    # 질문 임베딩
    try:
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=question)
        q_embedding = resp["embedding"]
    except Exception as e:
        return f"임베딩 오류: {e}\nollama pull {EMBED_MODEL} 을 먼저 실행하세요."

    # 유사 청크 검색
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"] or not results["documents"][0]:
        return "관련 문서를 찾지 못했습니다."

    # 컨텍스트 구성
    context_parts = []
    seen_sources = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        fname = meta.get("filename", "알 수 없음")
        context_parts.append(f"[출처: {fname}]\n{doc}")
        if fname not in seen_sources:
            seen_sources.append(fname)

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""다음은 문서에서 검색된 내용입니다:

{context}

위 내용을 바탕으로 다음 질문에 한국어로 답변하세요.
문서에 없는 내용은 "문서에서 찾을 수 없습니다"라고 답하세요.

질문: {question}

답변:"""

    try:
        response = ollama.generate(model=llm_model, prompt=prompt)
        answer = response["response"]
        sources = ", ".join(seen_sources)
        return f"{answer}\n\n[참조 파일: {sources}]"
    except Exception as e:
        return f"LLM 오류: {e}\nollama pull {llm_model} 을 실행했는지 확인하세요."


def list_indexed(db_path: str):
    client = get_chroma_client(db_path)
    try:
        collection = client.get_collection(COLLECTION_NAME)
        results = collection.get(include=["metadatas"])
        files = {}
        for meta in results["metadatas"]:
            if meta:
                src = meta.get("source", "?")
                files[src] = files.get(src, 0) + 1
        if not files:
            print("인덱싱된 파일이 없습니다.")
        else:
            print(f"\n인덱싱된 파일 ({len(files)}개):")
            for path, count in sorted(files.items()):
                print(f"  {Path(path).name} ({count}청크) - {path}")
    except Exception:
        print("인덱싱된 문서가 없습니다.")


def clear_index(db_path: str):
    client = get_chroma_client(db_path)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("인덱스가 초기화되었습니다.")
    except Exception:
        print("삭제할 인덱스가 없습니다.")


def interactive_mode(db_path: str, llm_model: str):
    print("\n" + "="*50)
    print("  Ollama RAG - 대화 모드")
    print("  종료: 'quit' 또는 'exit' 입력")
    print("  인덱스 목록: 'list'")
    print("="*50 + "\n")

    while True:
        try:
            question = input("질문> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "종료"):
            print("종료합니다.")
            break
        if question.lower() == "list":
            list_indexed(db_path)
            continue

        print("\n답변 생성 중...\n")
        answer = query(question, db_path, llm_model)
        print(answer)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Ollama RAG - 폴더 내 문서를 검색하고 질문에 답합니다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py index --folder C:\\문서폴더
  python main.py chat
  python main.py ask "보고서에서 매출 정보를 알려줘"
  python main.py list
  python main.py clear
        """
    )
    parser.add_argument("command", choices=["index", "chat", "ask", "list", "clear"],
                        help="실행할 명령")
    parser.add_argument("--folder", "-f", type=str,
                        help="문서 폴더 경로 (index 명령에 필요)")
    parser.add_argument("--question", "-q", type=str,
                        help="질문 (ask 명령에 사용)")
    parser.add_argument("--model", "-m", type=str, default="gemma3:4b",
                        help="사용할 LLM 모델 (기본값: gemma3:4b)")
    parser.add_argument("--db", type=str, default="./chroma_db",
                        help="벡터 DB 저장 경로 (기본값: ./chroma_db)")
    parser.add_argument("--force", action="store_true",
                        help="변경 없는 파일도 강제 재인덱싱")
    parser.add_argument("--top-k", type=int, default=5,
                        help="검색할 청크 수 (기본값: 5)")

    args = parser.parse_args()

    if args.command == "index":
        if not args.folder:
            parser.error("index 명령에는 --folder 옵션이 필요합니다.")
        folder = Path(args.folder)
        if not folder.exists():
            print(f"오류: 폴더를 찾을 수 없습니다: {folder}")
            sys.exit(1)
        index_documents(folder, args.db, args.force)

    elif args.command == "chat":
        interactive_mode(args.db, args.model)

    elif args.command == "ask":
        question = args.question
        if not question:
            # 나머지 인수를 질문으로 처리
            question = " ".join(sys.argv[3:])
        if not question:
            parser.error("ask 명령에는 질문이 필요합니다. --question 또는 위치 인수를 사용하세요.")
        print(f"\n질문: {question}\n")
        answer = query(question, args.db, args.model, args.top_k)
        print(answer)

    elif args.command == "list":
        list_indexed(args.db)

    elif args.command == "clear":
        confirm = input("인덱스를 초기화하시겠습니까? (y/N): ").strip().lower()
        if confirm == "y":
            clear_index(args.db)


if __name__ == "__main__":
    main()
