"""
Ollama RAG GUI - 문서 기반 AI 대화 프로그램
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import uuid
import sys
import hashlib
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    import ollama
    import chromadb
    from docx import Document as DocxDocument
    DEPS_OK = True
    DEPS_ERROR = None
except ImportError as e:
    DEPS_OK = False
    DEPS_ERROR = str(e)

try:
    from openai import OpenAI as _OpenAIClient
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

# 지원 제공자 정의
PROVIDERS = {
    "Ollama":     {"url": "http://localhost:11434", "port": 11434},
    "LM Studio":  {"url": "http://localhost:1234/v1", "port": 1234},
    "Jan":        {"url": "http://localhost:1337/v1", "port": 1337},
    "직접 입력":  {"url": "", "port": None},
}

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"
CONV_FILE  = BASE_DIR / "conversations.json"
DB_PATH    = str(BASE_DIR / "chroma_db")

CHUNK_SIZE     = 500
CHUNK_OVERLAP  = 100
EMBED_MODEL    = "nomic-embed-text"
COLLECTION     = "documents"

# ── 설정 ──────────────────────────────────────────────────────────────────────
def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"folder": "", "model": "gemma3:4b", "provider": "Ollama", "base_url": ""}

def save_config(cfg):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

# ── 대화 저장 ─────────────────────────────────────────────────────────────────
def load_conversations():
    if CONV_FILE.exists():
        with open(CONV_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_conversations(convs):
    with open(CONV_FILE, 'w', encoding='utf-8') as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)

# ── RAG 핵심 로직 ─────────────────────────────────────────────────────────────
def _read_txt(path):
    for enc in ["utf-8", "cp949", "euc-kr"]:
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return Path(path).read_text(encoding='utf-8', errors='replace')

def _read_docx(path):
    doc = DocxDocument(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def _read_rtf(path):
    """RTF 파일에서 순수 텍스트 추출 (외부 라이브러리 불필요)"""
    import re
    raw = Path(path).read_bytes()
    for enc in ["utf-8", "cp949", "euc-kr", "latin-1"]:
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    else:
        text = raw.decode("latin-1")

    # 유니코드 이스케이프 (\uN) 처리
    def _uni(m):
        try:
            return chr(int(m.group(1)))
        except Exception:
            return ""
    text = re.sub(r"\\u(-?\d+)\??", _uni, text)

    # 줄바꿈 제어어 → 개행
    text = re.sub(r"\\(par|line|pard)\b", "\n", text)
    # 탭
    text = re.sub(r"\\tab\b", "\t", text)
    # 특수문자 이스케이프 (\' 헥스)
    text = re.sub(r"\\'([0-9a-fA-F]{2})",
                  lambda m: bytes.fromhex(m.group(1)).decode("cp1252", errors="replace"), text)
    # 중괄호 그룹·제어어 제거
    text = re.sub(r"\{[^{}]*\}", "", text)
    text = re.sub(r"\\[a-zA-Z]+\-?\d*\s?", "", text)
    text = re.sub(r"[{}\\]", "", text)
    # 연속 공백 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def _file_hash(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()

def _ollama_client(base_url: str = ""):
    """base_url이 있으면 원격 Ollama 클라이언트, 없으면 로컬"""
    host = base_url.strip() or "http://localhost:11434"
    return ollama.Client(host=host)

def index_documents(folder, callback=None, force=False, ollama_url: str = ""):
    folder = Path(folder)
    docs = []
    for pat, reader in [("*.txt", _read_txt), ("*.docx", _read_docx), ("*.rtf", _read_rtf)]:
        for f in folder.rglob(pat):
            try:
                text = reader(f)
                if text.strip():
                    docs.append({"path": str(f), "text": text, "name": f.name})
            except Exception as e:
                if callback:
                    callback(f"  [건너뜀] {f.name}: {e}")
    if not docs:
        if callback:
            callback("지원 파일(.txt .docx .rtf)이 없습니다.")
        return 0, 0, 0

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing = set()
    try:
        for meta in col.get(include=["metadatas"])["metadatas"]:
            if meta and "file_hash" in meta:
                existing.add(meta["file_hash"])
    except Exception:
        pass

    added = skipped = errors = 0
    for doc in docs:
        fhash = _file_hash(doc["path"])
        if not force and fhash in existing:
            skipped += 1
            if callback:
                callback(f"  변경 없음: {doc['name']}")
            continue
        try:
            old = col.get(where={"source": doc["path"]})
            if old["ids"]:
                col.delete(ids=old["ids"])
        except Exception:
            pass

        chunks = _chunk(doc["text"])
        ok = True
        oc = _ollama_client(ollama_url)
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            try:
                emb = oc.embeddings(model=EMBED_MODEL, prompt=chunk)["embedding"]
                col.add(ids=[f"{fhash}_{i}"], embeddings=[emb], documents=[chunk],
                        metadatas=[{"source": doc["path"], "file_hash": fhash,
                                    "chunk_index": i, "filename": doc["name"]}])
                added += 1
            except Exception as e:
                errors += 1
                ok = False
                if callback:
                    callback(f"  [오류] {doc['name']} 청크{i}: {e}")
                break
        if ok and callback:
            callback(f"  완료: {doc['name']}")

    return added, skipped, errors

def _fetch_models_openai(base_url: str) -> list[str]:
    """LM Studio / Jan 등 OpenAI 호환 서버에서 모델 목록 조회"""
    try:
        c = _OpenAIClient(base_url=base_url, api_key="not-needed")
        return [m.id for m in c.models.list().data]
    except Exception:
        return []

def query_documents(question, model, top_k=5, provider="Ollama", base_url=""):
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        col = client.get_collection(COLLECTION)
        total = col.count()
        if total == 0:
            return "인덱싱된 문서가 없습니다. 설정 → 인덱싱을 먼저 실행하세요.", []
    except Exception:
        return "문서가 인덱싱되지 않았습니다. 설정 → 인덱싱을 먼저 실행하세요.", []

    # 전체 청크를 파일별로 묶어서 가져오기
    all_data = col.get(include=["documents", "metadatas"])
    file_chunks: dict[str, list[str]] = {}
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        fname = meta.get("filename", "?")
        idx   = meta.get("chunk_index", 0)
        if fname not in file_chunks:
            file_chunks[fname] = []
        file_chunks[fname].append((idx, doc))

    # 파일별로 청크를 순서대로 정렬 후 전체 텍스트 재구성
    ctx_parts, sources = [], []
    for fname, chunks in file_chunks.items():
        chunks.sort(key=lambda x: x[0])
        full_text = "\n".join(c[1] for c in chunks)
        ctx_parts.append(f"[파일: {fname}]\n{full_text}")
        sources.append(fname)

    prompt = (
        "당신은 아래 제공된 문서 내용만을 기반으로 답변하는 AI입니다.\n"
        "절대로 문서에 없는 내용을 추측하거나 일반 지식으로 답변하지 마세요.\n"
        "문서에서 답을 찾을 수 없으면 반드시 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고만 답하세요.\n\n"
        "=== 참조 문서 전체 ===\n\n"
        + "\n\n---\n\n".join(ctx_parts)
        + "\n\n=== 참조 문서 끝 ===\n\n"
        "위 문서 내용만 참고하여 아래 질문에 한국어로 상세하게 답변하세요.\n"
        "파일명이나 출처는 언급하지 말고 내용만 답변하세요.\n\n"
        f"질문: {question}\n\n답변:"
    )

    try:
        if provider == "Ollama":
            oc = _ollama_client(base_url)
            return oc.generate(model=model, prompt=prompt)["response"], sources
        else:
            # LM Studio / Jan / 직접 입력 — OpenAI 호환 API
            if not OPENAI_OK:
                return "openai 패키지가 없습니다. setup.bat 을 실행하세요.", []
            c = _OpenAIClient(base_url=base_url, api_key="not-needed")
            resp = c.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content":
                        "당신은 제공된 문서 내용만을 기반으로 답변하는 AI입니다. "
                        "문서에 없는 내용은 절대 추측하거나 일반 지식으로 답변하지 마세요. "
                        "문서에서 찾을 수 없는 내용은 '제공된 문서에서 해당 내용을 찾을 수 없습니다.'라고만 답하세요."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content, sources
    except Exception as e:
        return f"LLM 오류 ({provider}): {e}", []

# ── GUI ───────────────────────────────────────────────────────────────────────
C = {
    "sidebar":      "#1e1e2e",
    "sidebar_btn":  "#2a2a3e",
    "sidebar_sel":  "#313244",
    "main":         "#181825",
    "chat":         "#1e1e2e",
    "text":         "#cdd6f4",
    "text2":        "#a6adc8",
    "border":       "#45475a",
    "input_bg":     "#313244",
    "btn":          "#1d4ed8",
    "accent":       "#89b4fa",
    "ok":           "#a6e3a1",
    "err":          "#f38ba8",
    "warn":         "#fab387",
}


class OllamaRAGApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ollama RAG")
        self.root.geometry("1150x800")
        self.root.minsize(820, 600)
        self.root.configure(bg=C["main"])

        self.config = load_config()
        self.conversations = load_conversations()
        self.current_id: str | None = None
        self.busy = False

        self._build_ui()
        self._refresh_sidebar()

        if not DEPS_OK:
            messagebox.showerror("패키지 오류",
                                 f"필요한 패키지가 없습니다:\n{DEPS_ERROR}\n\n"
                                 "setup.bat 을 실행하세요.")

    # ── UI 구성 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        wrap = tk.Frame(self.root, bg=C["main"])
        wrap.pack(fill=tk.BOTH, expand=True)

        # ── 좌측 사이드바 ──
        self.sidebar = tk.Frame(wrap, bg=C["sidebar"], width=230)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # 앱 이름
        tk.Label(self.sidebar, text="Ollama RAG",
                 bg=C["sidebar"], fg=C["accent"],
                 font=("Malgun Gothic", 13, "bold"),
                 padx=14, pady=14).pack(anchor="w")

        # 새 대화 버튼
        tk.Button(self.sidebar, text="＋  새 대화",
                  bg=C["btn"], fg="white",
                  font=("Malgun Gothic", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=12, pady=7,
                  cursor="hand2",
                  command=self._new_conv).pack(fill=tk.X, padx=10, pady=(0, 8))

        tk.Frame(self.sidebar, bg=C["border"], height=1).pack(fill=tk.X, padx=10)

        # 대화 목록 스크롤 영역
        list_outer = tk.Frame(self.sidebar, bg=C["sidebar"])
        list_outer.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(list_outer, bg=C["sidebar"], highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(list_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)

        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.conv_frame = tk.Frame(canvas, bg=C["sidebar"])
        self._canvas_win = canvas.create_window((0, 0), window=self.conv_frame, anchor="nw")

        def _on_resize(e):
            canvas.itemconfig(self._canvas_win, width=e.width)
        canvas.bind("<Configure>", _on_resize)

        def _on_inner(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self.conv_frame.bind("<Configure>", _on_inner)

        # 마우스 휠 - 사이드바 영역에 마우스가 있을 때만 스크롤
        def _wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        def _bind_wheel(e):
            canvas.bind_all("<MouseWheel>", _wheel)

        def _unbind_wheel(e):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        self.conv_frame.bind("<Enter>", _bind_wheel)
        self.conv_frame.bind("<Leave>", _unbind_wheel)

        # 설정 버튼 (하단)
        tk.Frame(self.sidebar, bg=C["border"], height=1).pack(fill=tk.X, padx=10)
        tk.Button(self.sidebar, text="⚙   설정",
                  bg=C["sidebar"], fg=C["text2"],
                  font=("Malgun Gothic", 10),
                  relief=tk.FLAT, bd=0, padx=14, pady=9,
                  cursor="hand2", anchor="w",
                  activebackground=C["sidebar_btn"],
                  command=self._show_settings).pack(fill=tk.X, pady=(0, 4))

        # ── 우측 메인 영역 ──
        right = tk.Frame(wrap, bg=C["main"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 타이틀 바
        title_bar = tk.Frame(right, bg=C["main"], height=52)
        title_bar.pack(fill=tk.X)
        title_bar.pack_propagate(False)

        self.title_lbl = tk.Label(title_bar,
                                  text="대화를 선택하거나 새 대화를 시작하세요",
                                  bg=C["main"], fg=C["text"],
                                  font=("Malgun Gothic", 11, "bold"),
                                  padx=18)
        self.title_lbl.pack(side=tk.LEFT, pady=14)

        # 우측 상태 영역 (폴더 | 모델버튼 | 프로바이더)
        status_frame = tk.Frame(title_bar, bg=C["main"])
        status_frame.pack(side=tk.RIGHT, pady=14, padx=18)

        self.provider_lbl = tk.Label(status_frame, text="",
                                     bg=C["main"], fg=C["text2"],
                                     font=("Malgun Gothic", 9),
                                     cursor="hand2")
        self.provider_lbl.pack(side=tk.RIGHT)
        self.provider_lbl.bind("<Button-1>", lambda e: self._show_provider_picker())
        self.provider_lbl.bind("<Enter>", lambda e: self.provider_lbl.config(fg=C["accent"]))
        self.provider_lbl.bind("<Leave>", lambda e: self.provider_lbl.config(fg=C["text2"]))

        self.model_btn = tk.Label(status_frame, text="",
                                  bg=C["main"], fg=C["accent"],
                                  font=("Malgun Gothic", 9, "bold"),
                                  cursor="hand2", padx=6)
        self.model_btn.pack(side=tk.RIGHT)
        self.model_btn.bind("<Button-1>", lambda e: self._show_model_picker())
        self.model_btn.bind("<Enter>", lambda e: self.model_btn.config(fg=C["btn"]))
        self.model_btn.bind("<Leave>", lambda e: self.model_btn.config(fg=C["accent"]))

        self.folder_lbl = tk.Label(status_frame, text="",
                                   bg=C["main"], fg=C["text2"],
                                   font=("Malgun Gothic", 9),
                                   cursor="hand2")
        self.folder_lbl.pack(side=tk.RIGHT)
        self.folder_lbl.bind("<Button-1>", lambda e: self._pick_folder())
        self.folder_lbl.bind("<Enter>", lambda e: self.folder_lbl.config(fg=C["accent"]))
        self.folder_lbl.bind("<Leave>", lambda e: self.folder_lbl.config(fg=C["text2"]))

        self._refresh_status()

        tk.Frame(right, bg=C["border"], height=1).pack(fill=tk.X)

        # 채팅 표시 영역
        chat_wrap = tk.Frame(right, bg=C["chat"])
        chat_wrap.pack(fill=tk.BOTH, expand=True)

        self.chat = tk.Text(chat_wrap,
                            bg=C["chat"], fg=C["text"],
                            font=("Malgun Gothic", 10),
                            relief=tk.FLAT, bd=0,
                            padx=24, pady=20,
                            wrap=tk.WORD, state=tk.DISABLED,
                            cursor="arrow",
                            spacing1=2, spacing3=6)
        chat_sb = ttk.Scrollbar(chat_wrap, command=self.chat.yview)
        self.chat.configure(yscrollcommand=chat_sb.set)
        chat_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.chat.tag_configure("user_lbl",   foreground=C["accent"],
                                font=("Malgun Gothic", 9, "bold"))
        self.chat.tag_configure("user_msg",   foreground=C["text"],
                                font=("Malgun Gothic", 10),
                                lmargin1=28, lmargin2=28)
        self.chat.tag_configure("ai_lbl",     foreground=C["ok"],
                                font=("Malgun Gothic", 9, "bold"))
        self.chat.tag_configure("ai_msg",     foreground=C["text"],
                                font=("Malgun Gothic", 10),
                                lmargin1=28, lmargin2=28)
        self.chat.tag_configure("ai_src",     foreground=C["text2"],
                                font=("Malgun Gothic", 8),
                                lmargin1=28, lmargin2=28)
        self.chat.tag_configure("ai_err",     foreground=C["err"],
                                font=("Malgun Gothic", 10),
                                lmargin1=28, lmargin2=28)
        self.chat.tag_configure("system",     foreground=C["text2"],
                                font=("Malgun Gothic", 9, "italic"),
                                justify=tk.CENTER)
        self.chat.tag_configure("thinking",   foreground=C["warn"],
                                font=("Malgun Gothic", 9, "italic"),
                                lmargin1=28, lmargin2=28)

        self._show_welcome()

        # 입력 영역
        inp_outer = tk.Frame(right, bg=C["main"], padx=16, pady=12)
        inp_outer.pack(fill=tk.X, side=tk.BOTTOM)

        inp_box = tk.Frame(inp_outer, bg=C["input_bg"],
                           highlightthickness=1,
                           highlightbackground=C["border"],
                           highlightcolor=C["accent"])
        inp_box.pack(fill=tk.X)

        self.inp = tk.Text(inp_box,
                           bg=C["input_bg"], fg=C["text"],
                           font=("Malgun Gothic", 12),
                           relief=tk.FLAT, bd=0,
                           padx=12, pady=10, height=6,
                           wrap=tk.WORD,
                           insertbackground=C["text"])
        self.inp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inp.bind("<Return>", self._on_enter)
        self.inp.bind("<Shift-Return>", lambda e: None)
        self._set_placeholder(True)
        self.inp.bind("<FocusIn>",  lambda e: self._set_placeholder(False))
        self.inp.bind("<FocusOut>", lambda e: self._set_placeholder(
            not self.inp.get("1.0", tk.END).strip()))

        btn_col = tk.Frame(inp_box, bg=C["input_bg"])
        btn_col.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Frame(btn_col, bg=C["input_bg"]).pack(side=tk.TOP, expand=True)

        self.send_btn = tk.Button(btn_col, text="전송\n▶",
                                  bg=C["btn"], fg="white",
                                  font=("Malgun Gothic", 10, "bold"),
                                  relief=tk.FLAT, bd=0,
                                  padx=14, pady=12,
                                  cursor="hand2",
                                  command=self._send)
        self.send_btn.pack(side=tk.BOTTOM, padx=4, pady=4)

        tk.Label(inp_outer, text="Enter: 전송  │  Shift+Enter: 줄바꿈",
                 bg=C["main"], fg=C["text2"],
                 font=("Malgun Gothic", 8)).pack(anchor="e", pady=(4, 0))

    # ── 플레이스홀더 ─────────────────────────────────────────────────────────
    def _set_placeholder(self, show: bool):
        if show:
            self.inp.delete("1.0", tk.END)
            self.inp.insert("1.0", "질문을 입력하세요...")
            self.inp.config(fg=C["text2"])
        else:
            if self.inp.get("1.0", tk.END).strip() == "질문을 입력하세요...":
                self.inp.delete("1.0", tk.END)
            self.inp.config(fg=C["text"])

    def _on_enter(self, e):
        self._send()
        return "break"

    # ── 상태 바 ──────────────────────────────────────────────────────────────
    def _refresh_status(self):
        folder   = self.config.get("folder", "")
        model    = self.config.get("model", "gemma3:4b")
        provider = self.config.get("provider", "Ollama")
        if folder:
            self.folder_lbl.config(text=f"📁 {Path(folder).name}   ")
        else:
            self.folder_lbl.config(text="⚙ 설정 필요   ")
        self.model_btn.config(text=f"🤖 {model}   ")
        self.provider_lbl.config(text=f"🔌 {provider}")

    # ── 프로바이더 빠른 선택 팝업 ────────────────────────────────────────────
    def _show_provider_picker(self):
        popup = tk.Toplevel(self.root)
        popup.title("AI 제공자 선택")
        popup.configure(bg=C["main"])
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()

        bx = self.provider_lbl.winfo_rootx()
        by = self.provider_lbl.winfo_rooty() + self.provider_lbl.winfo_height() + 4
        popup.geometry(f"300x360+{bx}+{by}")

        tk.Label(popup, text="AI 제공자 선택", bg=C["main"], fg=C["accent"],
                 font=("Malgun Gothic", 10, "bold"), pady=10).pack()
        tk.Frame(popup, bg=C["border"], height=1).pack(fill=tk.X, padx=10)

        current_provider = self.config.get("provider", "Ollama")

        body = tk.Frame(popup, bg=C["main"])
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=10)

        selected_var = tk.StringVar(value=current_provider)
        url_var = tk.StringVar(value=self.config.get("base_url", ""))

        for name in PROVIDERS:
            row = tk.Frame(body, bg=C["main"])
            row.pack(fill=tk.X, pady=2)
            tk.Radiobutton(row, text=name, variable=selected_var, value=name,
                           bg=C["main"], fg=C["text"],
                           selectcolor=C["sidebar_btn"],
                           activebackground=C["main"], activeforeground=C["accent"],
                           font=("Malgun Gothic", 10),
                           anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 직접 입력 URL 필드
        tk.Frame(body, bg=C["border"], height=1).pack(fill=tk.X, pady=(8, 4))
        tk.Label(body, text="접속 URL (직접 입력 시 필수)", bg=C["main"], fg=C["text2"],
                 font=("Malgun Gothic", 8)).pack(anchor="w")
        url_entry = tk.Entry(body, textvariable=url_var,
                             bg=C["sidebar_btn"], fg=C["text"],
                             insertbackground=C["text"],
                             font=("Malgun Gothic", 9),
                             relief=tk.FLAT, bd=4)
        url_entry.pack(fill=tk.X, pady=(2, 8), ipady=3)

        def _on_select():
            pname = selected_var.get()
            url   = url_var.get().strip()
            if pname != "직접 입력":
                url = PROVIDERS[pname]["url"]
            self.config["provider"] = pname
            self.config["base_url"] = url
            # 프로바이더가 바뀌면 모델도 초기화
            self.config["model"] = ""
            save_config(self.config)
            self._refresh_status()
            popup.destroy()
            # 새 프로바이더의 기본 모델 자동 조회 후 첫 모델 선택
            self._auto_pick_first_model()

        tk.Frame(popup, bg=C["border"], height=1).pack(fill=tk.X, padx=10)
        tk.Button(popup, text="선택", bg=C["btn"], fg="white",
                  font=("Malgun Gothic", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=14, pady=6,
                  cursor="hand2", command=_on_select).pack(pady=8)

    def _auto_pick_first_model(self):
        """프로바이더 변경 후 첫 번째 모델을 자동으로 선택"""
        provider = self.config.get("provider", "Ollama")
        base_url = self.config.get("base_url", "").strip()
        url = base_url or PROVIDERS.get(provider, {}).get("url", "")

        def _fetch():
            models = []
            try:
                if provider == "Ollama":
                    models = [m.model for m in _ollama_client(url).list().models]
                elif OPENAI_OK and url:
                    models = _fetch_models_openai(url)
            except Exception:
                pass
            if models:
                self.root.after(0, lambda: self._set_model(models[0]))

        threading.Thread(target=_fetch, daemon=True).start()

    def _set_model(self, model: str):
        self.config["model"] = model
        save_config(self.config)
        self._refresh_status()

    # ── 폴더 빠른 선택 ──────────────────────────────────────────────────────
    def _pick_folder(self):
        path = filedialog.askdirectory(title="문서 폴더 선택", parent=self.root)
        if path:
            self.config["folder"] = path
            save_config(self.config)
            self._refresh_status()

    # ── 모델 빠른 선택 팝업 ──────────────────────────────────────────────────
    def _show_model_picker(self):
        provider = self.config.get("provider", "Ollama")
        base_url = self.config.get("base_url", "").strip()
        url = base_url or PROVIDERS.get(provider, {}).get("url", "")

        popup = tk.Toplevel(self.root)
        popup.title("모델 선택")
        popup.configure(bg=C["main"])
        popup.resizable(False, False)
        popup.transient(self.root)
        popup.grab_set()

        # 팝업 위치: 모델 버튼 아래
        bx = self.model_btn.winfo_rootx()
        by = self.model_btn.winfo_rooty() + self.model_btn.winfo_height() + 4
        popup.geometry(f"260x320+{bx}+{by}")

        tk.Label(popup, text="모델 선택", bg=C["main"], fg=C["accent"],
                 font=("Malgun Gothic", 10, "bold"), pady=10).pack()
        tk.Frame(popup, bg=C["border"], height=1).pack(fill=tk.X, padx=10)

        list_frame = tk.Frame(popup, bg=C["main"])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        loading_lbl = tk.Label(list_frame, text="모델 목록 조회 중…",
                               bg=C["main"], fg=C["text2"],
                               font=("Malgun Gothic", 9))
        loading_lbl.pack(pady=20)

        current_model = self.config.get("model", "")

        def _apply_models(models):
            loading_lbl.destroy()
            if not models:
                tk.Label(list_frame, text="설치된 모델이 없습니다",
                         bg=C["main"], fg=C["text2"],
                         font=("Malgun Gothic", 9)).pack(pady=20)
                return

            sb = ttk.Scrollbar(list_frame, orient="vertical")
            sb.pack(side=tk.RIGHT, fill=tk.Y)

            lb = tk.Listbox(list_frame,
                            bg=C["sidebar_btn"], fg=C["text"],
                            selectbackground=C["btn"],
                            selectforeground="white",
                            font=("Malgun Gothic", 10),
                            relief=tk.FLAT, bd=0,
                            activestyle="none",
                            yscrollcommand=sb.set)
            sb.config(command=lb.yview)
            lb.pack(fill=tk.BOTH, expand=True)

            for m in models:
                lb.insert(tk.END, m)
            if current_model in models:
                idx = models.index(current_model)
                lb.selection_set(idx)
                lb.see(idx)

            def _select(e=None):
                sel = lb.curselection()
                if not sel:
                    return
                chosen = lb.get(sel[0])
                self.config["model"] = chosen
                save_config(self.config)
                self._refresh_status()
                popup.destroy()

            lb.bind("<Double-Button-1>", _select)
            lb.bind("<Return>", _select)

            tk.Frame(popup, bg=C["border"], height=1).pack(fill=tk.X, padx=10)
            tk.Button(popup, text="선택", bg=C["btn"], fg="white",
                      font=("Malgun Gothic", 10, "bold"),
                      relief=tk.FLAT, bd=0, padx=14, pady=6,
                      cursor="hand2", command=_select).pack(pady=8)

        def _fetch():
            models = []
            try:
                if provider == "Ollama":
                    models = [m.model for m in _ollama_client(url).list().models]
                elif OPENAI_OK and url:
                    models = _fetch_models_openai(url)
            except Exception:
                pass
            popup.after(0, lambda: _apply_models(models))

        threading.Thread(target=_fetch, daemon=True).start()

    # ── 환영 메시지 ──────────────────────────────────────────────────────────
    def _show_welcome(self):
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)
        self.chat.insert(tk.END, "\n\n")
        self.chat.insert(tk.END, "Ollama RAG에 오신 것을 환영합니다!\n", "system")
        self.chat.insert(tk.END, "\n", "system")
        self.chat.insert(tk.END,
            "왼쪽 아래 ⚙ 설정에서 폴더를 지정하고 인덱싱하세요.\n", "system")
        self.chat.insert(tk.END,
            "그 후 ＋ 새 대화를 눌러 질문을 시작하세요.\n\n", "system")
        self.chat.config(state=tk.DISABLED)

    # ── 사이드바 갱신 ────────────────────────────────────────────────────────
    def _refresh_sidebar(self):
        for w in self.conv_frame.winfo_children():
            w.destroy()

        if not self.conversations:
            tk.Label(self.conv_frame, text="대화가 없습니다",
                     bg=C["sidebar"], fg=C["text2"],
                     font=("Malgun Gothic", 9)).pack(pady=12)
            return

        for conv in reversed(self.conversations):
            self._add_conv_row(conv)

    def _add_conv_row(self, conv: dict):
        sel  = conv["id"] == self.current_id
        bg   = C["sidebar_sel"] if sel else C["sidebar"]
        fgc  = C["text"] if sel else C["text2"]
        title = conv.get("title", "새 대화")
        if len(title) > 22:
            title = title[:22] + "…"

        row = tk.Frame(self.conv_frame, bg=bg, cursor="hand2")
        row.pack(fill=tk.X, padx=4, pady=1)

        lbl = tk.Label(row, text=f"💬  {title}",
                       bg=bg, fg=fgc,
                       font=("Malgun Gothic", 10),
                       anchor="w", padx=8, pady=7)
        lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        del_btn = tk.Button(row, text="✕",
                            bg=bg, fg=C["text2"],
                            font=("Malgun Gothic", 9),
                            relief=tk.FLAT, bd=0,
                            padx=6, pady=5,
                            cursor="hand2",
                            command=lambda cid=conv["id"]: self._delete_conv(cid))
        del_btn.pack(side=tk.RIGHT, padx=(0, 4))

        cid = conv["id"]
        for w in (row, lbl):
            w.bind("<Button-1>", lambda e, c=cid: self._load_conv(c))

        def _enter(e, widgets=(row, lbl, del_btn), c=cid):
            if c != self.current_id:
                for w in widgets:
                    w.config(bg=C["sidebar_btn"])

        def _leave(e, widgets=(row, lbl, del_btn), c=cid):
            nbg = C["sidebar_sel"] if c == self.current_id else C["sidebar"]
            for w in widgets:
                w.config(bg=nbg)

        for w in (row, lbl, del_btn):
            w.bind("<Enter>", _enter)
            w.bind("<Leave>", _leave)

    # ── 대화 관리 ────────────────────────────────────────────────────────────
    def _new_conv(self):
        conv = {"id": str(uuid.uuid4()),
                "title": "새 대화",
                "created_at": datetime.now().isoformat(),
                "messages": []}
        self.conversations.append(conv)
        save_conversations(self.conversations)
        self.current_id = conv["id"]
        self._refresh_sidebar()
        self._render_conv(conv)

    def _load_conv(self, cid: str):
        self.current_id = cid
        self._refresh_sidebar()
        conv = self._find(cid)
        if conv:
            self._render_conv(conv)

    def _delete_conv(self, cid: str):
        self.conversations = [c for c in self.conversations if c["id"] != cid]
        save_conversations(self.conversations)
        if self.current_id == cid:
            self.current_id = None
            self.title_lbl.config(text="대화를 선택하거나 새 대화를 시작하세요")
            self._show_welcome()
        self._refresh_sidebar()

    def _find(self, cid: str) -> dict | None:
        return next((c for c in self.conversations if c["id"] == cid), None)

    def _render_conv(self, conv: dict):
        self.title_lbl.config(text=conv.get("title", "새 대화"))
        self.chat.config(state=tk.NORMAL)
        self.chat.delete("1.0", tk.END)

        if not conv["messages"]:
            self.chat.insert(tk.END, "\n\n")
            self.chat.insert(tk.END, "무엇이든 물어보세요.\n", "system")

        for msg in conv["messages"]:
            self._insert_msg(msg["role"], msg["content"], msg.get("sources", []))

        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _insert_msg(self, role: str, content: str, sources: list):
        if role == "user":
            self.chat.insert(tk.END, "\n👤  나\n", "user_lbl")
            self.chat.insert(tk.END, content + "\n", "user_msg")
        else:
            self.chat.insert(tk.END, "\n🤖  AI\n", "ai_lbl")
            is_err = any(content.startswith(p)
                         for p in ("오류", "LLM 오류", "임베딩 오류",
                                   "문서가", "인덱싱된", "관련 문서"))
            self.chat.insert(tk.END, content + "\n",
                             "ai_err" if is_err else "ai_msg")
            if sources:
                self.chat.insert(tk.END,
                                 f"📎 {', '.join(sources)}\n", "ai_src")

    # ── 메시지 전송 ──────────────────────────────────────────────────────────
    def _send(self):
        if self.busy:
            return
        question = self.inp.get("1.0", tk.END).strip()
        if not question or question == "질문을 입력하세요...":
            return
        if not self.config.get("folder"):
            messagebox.showinfo("안내",
                "먼저 ⚙ 설정에서 문서 폴더를 지정하고 인덱싱하세요.")
            return

        if not self.current_id:
            self._new_conv()

        conv = self._find(self.current_id)
        if not conv:
            return

        self.inp.delete("1.0", tk.END)
        self.inp.config(fg=C["text"])

        conv["messages"].append({"role": "user", "content": question})
        if len(conv["messages"]) == 1:
            conv["title"] = question[:30] + ("…" if len(question) > 30 else "")
            self.title_lbl.config(text=conv["title"])
        save_conversations(self.conversations)
        self._refresh_sidebar()

        self.chat.config(state=tk.NORMAL)
        self._insert_msg("user", question, [])
        self.chat.insert(tk.END, "\n🤖  AI\n", "ai_lbl")
        self.chat.mark_set("thinking_start", "end-1c")
        self.chat.insert(tk.END, "생각하는 중...\n", "thinking")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

        self.busy = True
        self.send_btn.config(state=tk.DISABLED, text="처리 중…")

        model    = self.config.get("model", "gemma3:4b")
        provider = self.config.get("provider", "Ollama")
        base_url = self.config.get("base_url", "")
        cid      = self.current_id

        def _worker():
            answer, sources = query_documents(question, model,
                                              provider=provider, base_url=base_url)
            self.root.after(0, lambda: self._on_answer(cid, answer, sources))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_answer(self, cid: str, answer: str, sources: list):
        self.busy = False
        self.send_btn.config(state=tk.NORMAL, text="전송  ▶")

        conv = self._find(cid)
        if not conv:
            return
        conv["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources})
        save_conversations(self.conversations)

        self.chat.config(state=tk.NORMAL)
        # "생각하는 중..." 삭제
        self.chat.delete("thinking_start", tk.END)
        self._insert_msg("assistant", answer, sources)
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    # ── 설정 다이얼로그 ──────────────────────────────────────────────────────
    def _show_settings(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("설정")
        dlg.geometry("560x620")
        dlg.configure(bg=C["main"])
        dlg.resizable(False, False)
        dlg.transient(self.root)
        dlg.grab_set()

        dlg.update_idletasks()
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        dlg.geometry(f"560x620+{rx + (rw-560)//2}+{ry + (rh-620)//2}")

        tk.Label(dlg, text="설정",
                 bg=C["main"], fg=C["text"],
                 font=("Malgun Gothic", 14, "bold")).pack(
            anchor="w", padx=22, pady=(20, 14))

        body = tk.Frame(dlg, bg=C["main"])
        body.pack(fill=tk.BOTH, expand=True, padx=22)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TCombobox",
                        fieldbackground=C["input_bg"],
                        background=C["input_bg"],
                        foreground=C["text"],
                        arrowcolor=C["text"])

        def _lbl(parent, text):
            tk.Label(parent, text=text, bg=C["main"], fg=C["text2"],
                     font=("Malgun Gothic", 9, "bold")).pack(anchor="w", pady=(0, 4))

        def _entry(parent, var, **kw):
            e = tk.Entry(parent, textvariable=var,
                         bg=C["input_bg"], fg=C["text"],
                         insertbackground=C["text"],
                         font=("Malgun Gothic", 10),
                         relief=tk.FLAT, bd=4, **kw)
            return e

        def _combo(parent, var, values):
            cb = ttk.Combobox(parent, textvariable=var, values=values,
                              font=("Malgun Gothic", 10),
                              state="readonly", style="Dark.TCombobox")
            cb.pack(fill=tk.X, pady=(0, 14), ipady=4)
            return cb

        # ── 폴더 선택 ──
        _lbl(body, "문서 폴더")
        folder_row = tk.Frame(body, bg=C["main"])
        folder_row.pack(fill=tk.X, pady=(0, 14))
        folder_var = tk.StringVar(value=self.config.get("folder", ""))
        _entry(folder_row, folder_var).pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4)

        def _browse():
            p = filedialog.askdirectory(title="문서 폴더 선택", parent=dlg)
            if p:
                folder_var.set(p)

        tk.Button(folder_row, text="찾아보기",
                  bg=C["btn"], fg="white", font=("Malgun Gothic", 9),
                  relief=tk.FLAT, bd=0, padx=10, pady=4,
                  cursor="hand2", command=_browse).pack(side=tk.LEFT, padx=(6, 0))

        tk.Frame(body, bg=C["border"], height=1).pack(fill=tk.X, pady=(0, 14))

        # ── AI 제공자 선택 ──
        _lbl(body, "AI 제공자")
        provider_var = tk.StringVar(value=self.config.get("provider", "Ollama"))
        _combo(body, provider_var, list(PROVIDERS.keys()))

        # 접속 URL
        _lbl(body, "접속 URL")
        url_var = tk.StringVar(value=self.config.get("base_url", ""))
        url_ent = _entry(body, url_var)
        url_ent.pack(fill=tk.X, pady=(0, 4), ipady=4)
        tk.Label(body, text="Ollama: http://localhost:11434  |  LM Studio: http://localhost:1234/v1  |  Jan: http://localhost:1337/v1",
                 bg=C["main"], fg=C["text2"],
                 font=("Malgun Gothic", 8),
                 wraplength=500, justify=tk.LEFT).pack(anchor="w", pady=(0, 14))

        def _on_provider_change(*_):
            pname = provider_var.get()
            preset_url = PROVIDERS.get(pname, {}).get("url", "")
            if preset_url:
                url_var.set(preset_url)
            _refresh_models()

        provider_var.trace_add("write", _on_provider_change)

        # ── 모델 선택 ──
        _lbl(body, "LLM 모델")
        model_var = tk.StringVar(value=self.config.get("model", ""))
        model_cb  = ttk.Combobox(body, textvariable=model_var,
                                  font=("Malgun Gothic", 10),
                                  style="Dark.TCombobox")
        model_cb.pack(fill=tk.X, pady=(0, 4), ipady=4)

        refresh_btn = tk.Button(body, text="↺  모델 목록 새로고침",
                                bg=C["sidebar_btn"], fg=C["text2"],
                                font=("Malgun Gothic", 9),
                                relief=tk.FLAT, bd=0, padx=10, pady=4,
                                cursor="hand2")

        def _refresh_models():
            refresh_btn.config(state=tk.DISABLED, text="조회 중…")
            pname = provider_var.get()
            url   = url_var.get().strip() or PROVIDERS.get(pname, {}).get("url", "")

            def _fetch():
                models = []
                try:
                    if pname == "Ollama":
                        models = [m.model for m in _ollama_client(url).list().models]
                    elif OPENAI_OK and url:
                        models = _fetch_models_openai(url)
                except Exception:
                    pass
                dlg.after(0, lambda m=models: _apply_models(m))

            def _apply_models(models):
                refresh_btn.config(state=tk.NORMAL, text="↺  모델 목록 새로고침")
                if models:
                    model_cb.config(values=models, state="readonly")
                    if model_var.get() not in models:
                        model_var.set(models[0])
                else:
                    model_cb.config(values=[], state="normal")

            threading.Thread(target=_fetch, daemon=True).start()

        refresh_btn.config(command=_refresh_models)
        refresh_btn.pack(anchor="w", pady=(0, 14))

        # 초기 모델 목록 로드
        _on_provider_change()

        tk.Frame(body, bg=C["border"], height=1).pack(fill=tk.X, pady=(0, 14))

        # ── 인덱싱 ──
        _lbl(body, "문서 인덱싱")
        log_box = tk.Text(body, bg=C["input_bg"], fg=C["text"],
                          font=("Malgun Gothic", 9),
                          height=5, relief=tk.FLAT, bd=4,
                          state=tk.DISABLED)
        log_box.pack(fill=tk.X, pady=(0, 8))

        idx_btn = tk.Button(body, text="▶  인덱싱 시작",
                            bg=C["btn"], fg="white",
                            font=("Malgun Gothic", 10, "bold"),
                            relief=tk.FLAT, bd=0, padx=14, pady=5,
                            cursor="hand2")
        idx_btn.pack(anchor="w")

        def _log(msg):
            dlg.after(0, lambda m=msg: (
                log_box.config(state=tk.NORMAL),
                log_box.insert(tk.END, m + "\n"),
                log_box.see(tk.END),
                log_box.config(state=tk.DISABLED)
            ))

        def _do_index():
            folder = folder_var.get().strip()
            if not folder:
                messagebox.showwarning("경고", "폴더를 먼저 선택하세요.", parent=dlg)
                return
            _apply_config()
            idx_btn.config(state=tk.DISABLED, text="인덱싱 중…")
            log_box.config(state=tk.NORMAL)
            log_box.delete("1.0", tk.END)
            log_box.config(state=tk.DISABLED)

            def _run():
                _log(f"폴더 스캔 중: {folder}")
                pname = provider_var.get()
                ourl  = url_var.get().strip() if pname == "Ollama" else ""
                added, skipped, errors = index_documents(folder, callback=_log,
                                                         ollama_url=ourl)
                _log(f"\n✅ 완료 — {added}청크 추가, {skipped}파일 생략, {errors}오류")
                dlg.after(0, lambda: idx_btn.config(state=tk.NORMAL,
                                                    text="▶  인덱싱 시작"))

            threading.Thread(target=_run, daemon=True).start()

        idx_btn.config(command=_do_index)

        # ── 하단 버튼 ──
        btn_row = tk.Frame(dlg, bg=C["main"])
        btn_row.pack(fill=tk.X, padx=22, pady=16)

        def _apply_config():
            self.config["folder"]   = folder_var.get().strip()
            self.config["provider"] = provider_var.get()
            self.config["base_url"] = url_var.get().strip() or \
                                      PROVIDERS.get(provider_var.get(), {}).get("url", "")
            self.config["model"]    = model_var.get()
            save_config(self.config)
            self._refresh_status()

        def _save_close():
            _apply_config()
            dlg.destroy()

        tk.Button(btn_row, text="저장 후 닫기",
                  bg=C["btn"], fg="white",
                  font=("Malgun Gothic", 10, "bold"),
                  relief=tk.FLAT, bd=0, padx=14, pady=6,
                  cursor="hand2", command=_save_close).pack(side=tk.RIGHT)

        tk.Button(btn_row, text="닫기",
                  bg=C["sidebar_btn"], fg=C["text"],
                  font=("Malgun Gothic", 10),
                  relief=tk.FLAT, bd=0, padx=14, pady=6,
                  cursor="hand2", command=dlg.destroy).pack(side=tk.RIGHT, padx=(0, 8))


def main():
    root = tk.Tk()
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    OllamaRAGApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
