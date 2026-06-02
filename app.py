import argparse
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
DEFAULT_EMAIL_DIR = ROOT / "emails"
STATIC_DIR = ROOT / "static"
MODEL_NAME = "BAAI/bge-m3"
DEFAULT_CHUNK_TOKENS = int(os.environ.get("EMAIL_RAG_DEFAULT_CHUNK_TOKENS", "512"))
DEFAULT_CHUNK_OVERLAP_TOKENS = 64
DEFAULT_DB = ROOT / f"email_rag_{DEFAULT_CHUNK_TOKENS}.sqlite3"
DEFAULT_SEARCH_INDEX_TARGET = "all"
TITLE_DB = ROOT / "rag_title.sqlite3"
CONTENT_DB = ROOT / "rag_content.sqlite3"
EMAIL_DB = ROOT / "rag_email.sqlite3"
ACTIVE_DB = DEFAULT_DB

app = FastAPI(title="Email RAG")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
_MODEL = None
_TOKENIZER = None


@dataclass
class ParsedEmail:
    source_path: str
    file_mtime: float
    message_id: str
    subject: str
    from_addr: str
    to_addr: str
    date: str
    date_sort: str
    body: str


class SearchRequest(BaseModel):
    query: str
    limit: int = 8
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    index_target: str = DEFAULT_SEARCH_INDEX_TARGET


class ChatRequest(BaseModel):
    message: str
    limit: int = 8
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    index_target: str = DEFAULT_SEARCH_INDEX_TARGET
class SearchPlan(BaseModel):
    keywords: list[str] = []
    date_from: str | None = None
    date_to: str | None = None
    sender: str | None = None


def normalize_chunk_tokens(chunk_tokens: int) -> int:
    if chunk_tokens not in {256, 512}:
        raise HTTPException(status_code=400, detail="chunk_tokens must be 256 or 512")
    return chunk_tokens


def db_path_for_chunk_tokens(chunk_tokens: int) -> Path:
    return ROOT / f"email_rag_{normalize_chunk_tokens(chunk_tokens)}.sqlite3"


def db_path_for_index_target(index_target: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Path:
    if index_target == "chunk":
        return db_path_for_chunk_tokens(chunk_tokens)
    if index_target == "title":
        return TITLE_DB
    if index_target == "content":
        return CONTENT_DB
    if index_target == "email":
        return EMAIL_DB
    raise ValueError(f"unknown index target: {index_target}")




def normalize_search_index_target(index_target: str) -> str:
    if index_target == "all":
        return "email"
    if index_target in {"chunk", "title", "content", "email"}:
        return index_target
    raise HTTPException(status_code=400, detail="index_target must be chunk, title, content, email, or all")


def db_path_for_search_target(index_target: str, chunk_tokens: int = DEFAULT_CHUNK_TOKENS) -> Path:
    return db_path_for_index_target(normalize_search_index_target(index_target), chunk_tokens)


def set_active_db(db_path: Path):
    global ACTIVE_DB
    ACTIVE_DB = db_path


def connect(db_path: Path | None = None):
    conn = sqlite3.connect(db_path or ACTIVE_DB)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn):
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_path TEXT NOT NULL UNIQUE,
            file_mtime REAL NOT NULL,
            message_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            from_addr TEXT NOT NULL,
            to_addr TEXT NOT NULL,
            date TEXT NOT NULL,
            date_sort TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            UNIQUE(email_id, chunk_index)
        );

        CREATE INDEX IF NOT EXISTS idx_emails_date_sort ON emails(date_sort DESC);
        CREATE INDEX IF NOT EXISTS idx_chunks_email_id ON chunks(email_id);
    ''')
    conn.commit()


def parse_email_file(path: Path) -> ParsedEmail:
    text = path.read_text(encoding="utf-8", errors="replace")
    header_text, _, body = text.partition("\n\n")
    headers = {}
    for line in header_text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            headers[key.strip().lower()] = value.strip()

    message_id = headers.get("gmail message id") or path.stem.rsplit("_", 1)[-1]
    date_value = headers.get("date", "")
    date_sort = normalize_date(date_value)
    subject = headers.get("subject") or path.stem

    return ParsedEmail(
        source_path=str(path),
        file_mtime=path.stat().st_mtime,
        message_id=message_id,
        subject=subject,
        from_addr=headers.get("from", ""),
        to_addr=headers.get("to", ""),
        date=date_value,
        date_sort=date_sort,
        body=body.strip(),
    )


def normalize_date(value: str) -> str:
    if not value:
        return ""
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.isoformat()
    except Exception:
        return value


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _TOKENIZER


def chunk_text(
    text: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return [""]
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")

    overlap_tokens = max(0, min(overlap_tokens, chunk_tokens - 1))
    tok = tokenizer()
    token_ids = tok.encode(normalized, add_special_tokens=False)
    if not token_ids:
        return [normalized]

    chunks = []
    start = 0
    while start < len(token_ids):
        end = min(len(token_ids), start + chunk_tokens)
        chunk_ids = token_ids[start:end]
        chunk = tok.decode(chunk_ids, skip_special_tokens=True).strip()
        while chunk_ids and len(tok.encode(chunk, add_special_tokens=False)) > chunk_tokens:
            chunk_ids = chunk_ids[:-1]
            end = start + len(chunk_ids)
            chunk = tok.decode(chunk_ids, skip_special_tokens=True).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(token_ids):
            break
        start = max(start + 1, end - overlap_tokens)
    return chunks


def model():
    global _MODEL
    if _MODEL is None:
        from FlagEmbedding import BGEM3FlagModel

        _MODEL = BGEM3FlagModel(MODEL_NAME, use_fp16=True, return_dense=True)
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    vectors = model().encode(
        texts,
        batch_size=batch_size,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def blob_to_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def email_index_text(email: ParsedEmail) -> str:
    return "\n".join([
        f"Subject: {email.subject}",
        f"From: {email.from_addr}",
        f"To: {email.to_addr}",
        f"Date: {email.date}",
        "",
        email.body,
    ]).strip()


def upsert_email(conn, email: ParsedEmail) -> tuple[int, bool]:
    existing = conn.execute(
        "SELECT id, file_mtime FROM emails WHERE source_path = ?",
        (email.source_path,),
    ).fetchone()
    if existing and abs(float(existing["file_mtime"]) - email.file_mtime) < 0.001:
        chunk_count = conn.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE email_id = ?",
            (existing["id"],),
        ).fetchone()["count"]
        if chunk_count:
            return int(existing["id"]), False

    if existing:
        email_id = int(existing["id"])
        conn.execute('''
            UPDATE emails
            SET file_mtime = ?, message_id = ?, subject = ?, from_addr = ?,
                to_addr = ?, date = ?, date_sort = ?, body = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            email.file_mtime,
            email.message_id,
            email.subject,
            email.from_addr,
            email.to_addr,
            email.date,
            email.date_sort,
            email.body,
            email_id,
        ))
        conn.execute("DELETE FROM chunks WHERE email_id = ?", (email_id,))
        return email_id, True

    cursor = conn.execute('''
        INSERT INTO emails (
            source_path, file_mtime, message_id, subject, from_addr, to_addr, date, date_sort, body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        email.source_path,
        email.file_mtime,
        email.message_id,
        email.subject,
        email.from_addr,
        email.to_addr,
        email.date,
        email.date_sort,
        email.body,
    ))
    return int(cursor.lastrowid), True


def chunks_for_index_target(
    email: ParsedEmail,
    index_target: str,
    chunk_tokens: int,
    chunk_overlap_tokens: int,
) -> list[str]:
    if index_target == "title":
        title = email.subject.strip() or email.message_id
        return [title]
    if index_target == "content":
        return chunk_text(
            email.body,
            chunk_tokens=chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )
    if index_target == "email":
        return ["\n\n".join([
            f"Subject: {email.subject}",
            email.body,
        ]).strip()]
    if index_target == "chunk":
        return chunk_text(
            email_index_text(email),
            chunk_tokens=chunk_tokens,
            overlap_tokens=chunk_overlap_tokens,
        )
    raise ValueError(f"unknown index target: {index_target}")


def build_index(
    email_dir: Path = DEFAULT_EMAIL_DIR,
    db_path: Path | None = None,
    reindex: bool = False,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    index_target: str = "chunk",
):
    normalize_chunk_tokens(chunk_tokens)
    db_path = db_path or db_path_for_index_target(index_target, chunk_tokens)
    conn = connect(db_path)
    init_db(conn)
    if reindex:
        conn.executescript("DELETE FROM chunks; DELETE FROM emails;")
        conn.commit()

    paths = sorted(email_dir.glob("*.txt"))
    pending: list[tuple[int, ParsedEmail, list[str]]] = []

    for path in paths:
        parsed = parse_email_file(path)
        email_id, needs_embedding = upsert_email(conn, parsed)
        if needs_embedding:
            chunks = chunks_for_index_target(
                parsed,
                index_target=index_target,
                chunk_tokens=chunk_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
            pending.append((email_id, parsed, chunks))

    conn.commit()
    total_chunks = sum(len(item[2]) for item in pending)
    print(f"index target: {index_target}")
    print(f"db: {db_path}")
    if index_target not in {"title", "email"}:
        print(f"chunk tokens: {chunk_tokens}")
        print(f"chunk overlap tokens: {chunk_overlap_tokens}")
    print(f"emails scanned: {len(paths)}")
    print(f"emails needing embeddings: {len(pending)}")
    print(f"vectors needing embeddings: {total_chunks}")

    done_chunks = 0
    for email_id, _parsed, chunks in pending:
        vectors = embed_texts(chunks)
        for chunk_index, (chunk_content, vector) in enumerate(zip(chunks, vectors)):
            conn.execute('''
                INSERT INTO chunks (email_id, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
            ''', (email_id, chunk_index, chunk_content, vector_to_blob(vector)))
        done_chunks += len(chunks)
        if done_chunks % 25 == 0 or done_chunks == total_chunks:
            print(f"embedded vectors: {done_chunks}/{total_chunks}", flush=True)
        conn.commit()

    conn.close()


def normalize_query_date(value: str | None, end_of_day: bool = False) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return f"{value}T23:59:59" if end_of_day else f"{value}T00:00:00"
    return value


def sql_filters_from_plan(plan: dict):
    clauses = []
    params = []

    sender = (plan.get("sender") or "").strip()
    if sender:
        clauses.append("emails.from_addr LIKE ?")
        params.append(f"%{sender}%")

    date_from = normalize_query_date(plan.get("date_from"))
    if date_from:
        clauses.append("emails.date_sort >= ?")
        params.append(date_from)

    date_to = normalize_query_date(plan.get("date_to"), end_of_day=True)
    if date_to:
        clauses.append("emails.date_sort <= ?")
        params.append(date_to)

    keywords = [str(keyword).strip() for keyword in plan.get("keywords", []) if str(keyword).strip()]
    for keyword in keywords:
        like = f"%{keyword}%"
        clauses.append("(emails.subject LIKE ? OR emails.body LIKE ? OR emails.from_addr LIKE ? OR emails.to_addr LIKE ?)")
        params.extend([like, like, like, like])

    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    return where, params


def search_emails(
    query: str,
    limit: int = 8,
    db_path: Path | None = None,
    plan: dict | None = None,
):
    if not query.strip():
        return []
    conn = connect(db_path)
    init_db(conn)
    where, params = sql_filters_from_plan(plan or {})
    rows = conn.execute(f'''
        SELECT chunks.id AS chunk_id, chunks.email_id, chunks.chunk_index, chunks.content,
               chunks.embedding, emails.subject, emails.from_addr, emails.to_addr,
               emails.date, emails.date_sort
        FROM chunks
        JOIN emails ON emails.id = chunks.email_id
        {where}
    ''', params).fetchall()
    if not rows:
        conn.close()
        return []

    query_vector = embed_texts([query])[0]
    best_by_email = {}
    for row in rows:
        vector = blob_to_vector(row["embedding"])
        score = float(np.dot(query_vector, vector))
        current = best_by_email.get(row["email_id"])
        if current is None or score > current["score"]:
            best_by_email[row["email_id"]] = {
                "id": row["email_id"],
                "subject": row["subject"],
                "from": row["from_addr"],
                "to": row["to_addr"],
                "date": row["date"],
                "date_sort": row["date_sort"],
                "score": score,
                "chunk": row["content"],
            }

    results = sorted(best_by_email.values(), key=lambda item: item["score"], reverse=True)[:limit]
    conn.close()
    return results

def get_email(email_id: int, db_path: Path | None = None):
    conn = connect(db_path)
    init_db(conn)
    row = conn.execute("SELECT * FROM emails WHERE id = ?", (email_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    return dict(row)


def list_emails(limit: int = 100, offset: int = 0, db_path: Path | None = None):
    conn = connect(db_path)
    init_db(conn)
    rows = conn.execute('''
        SELECT id, subject, from_addr AS 'from', to_addr AS 'to', date, date_sort
        FROM emails
        ORDER BY date_sort DESC, id DESC
        LIMIT ? OFFSET ?
    ''', (limit, offset)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def build_rag_prompt(question: str, results: list[dict]) -> str:
    sources = []
    for index, item in enumerate(results, start=1):
        sources.append("\n".join([
            f"[Email {index}]",
            f"Subject: {item['subject']}",
            f"From: {item['from']}",
            f"To: {item['to']}",
            f"Date: {item['date']}",
            f"Score: {item['score']:.4f}",
            "Content:",
            item["chunk"][:3500],
        ]))
    context = "\n\n---\n\n".join(sources)
    return (
        "You are answering questions using only the retrieved email context below.\n"
        "Answer in Korean unless the user asks otherwise.\n"
        "If the answer is not supported by the emails, say that the retrieved emails do not contain enough information.\n"
        "Cite relevant emails by subject and date.\n"
        "Do not invent facts beyond the email context.\n\n"
        f"Retrieved emails:\n{context}\n\n"
        f"User question:\n{question}\n"
    )


def extract_json_object(value: str) -> dict:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    start = value.find("{")
    end = value.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        parsed = json.loads(value[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def normalize_search_plan(raw: dict, question: str) -> dict:
    keywords = raw.get("keywords") if isinstance(raw, dict) else []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        keywords = []
    keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if not keywords:
        keywords = [question.strip()]

    plan = {
        "keywords": keywords[:8],
        "date_from": raw.get("date_from") or None,
        "date_to": raw.get("date_to") or None,
        "sender": raw.get("sender") or None,
    }
    for key in ("date_from", "date_to", "sender"):
        if isinstance(plan[key], str):
            plan[key] = plan[key].strip() or None
    return plan


def extract_search_plan(question: str) -> dict:
    today = datetime.now().date().isoformat()
    prompt = (
        "Extract an email search plan from the Korean user question. "
        "Return only compact JSON with keys: keywords, date_from, date_to, sender. "
        "keywords must be a list of short search terms suitable for SQL LIKE search. "
        "date_from/date_to must be YYYY-MM-DD or null. sender is a sender name or email substring, or null. "
        "Use sender only when the question explicitly names a sender/person/organization/email. "
        "Use the current date only to resolve relative dates. "
        f"Current date: {today}.\n"
        f"Question: {question}\n"
    )
    raw_answer = run_codex_exec(prompt, timeout=120)
    parsed = extract_json_object(raw_answer)
    return normalize_search_plan(parsed, question)

def run_codex_exec(prompt: str, timeout: int = 240) -> str:
    with tempfile.NamedTemporaryFile(prefix="email_rag_codex_", suffix=".txt", delete=False) as tmp:
        output_path = tmp.name
    try:
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--output-last-message",
                output_path,
                "-",
            ],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(ROOT),
            check=False,
        )
        answer = Path(output_path).read_text(encoding="utf-8", errors="replace").strip()
        if answer:
            return answer
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        if completed.returncode != 0:
            return f"codex exec failed with exit code {completed.returncode}.\n{stderr or stdout}"
        return stdout or "codex exec returned no output."
    except subprocess.TimeoutExpired:
        return "codex exec timed out before producing an answer."
    finally:
        try:
            os.unlink(output_path)
        except OSError:
            pass


@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


def index_db_stats(db_path: Path) -> dict:
    email_count = 0
    vector_count = 0
    if db_path.exists():
        conn = connect(db_path)
        init_db(conn)
        email_count = conn.execute("SELECT COUNT(*) AS count FROM emails").fetchone()["count"]
        vector_count = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        conn.close()
    return {
        "db": db_path.name,
        "exists": db_path.exists(),
        "emails": email_count,
        "chunks": vector_count,
    }


@app.get("/api/indexes")
def api_indexes():
    indexes = []
    for chunk_tokens in (256, 512):
        db_path = db_path_for_chunk_tokens(chunk_tokens)
        item = index_db_stats(db_path)
        item.update({"index_target": "chunk", "chunk_tokens": chunk_tokens})
        indexes.append(item)
    for target, db_path in (("title", TITLE_DB), ("content", CONTENT_DB), ("email", EMAIL_DB)):
        item = index_db_stats(db_path)
        item.update({"index_target": target, "chunk_tokens": None})
        indexes.append(item)
    return {"default_chunk_tokens": DEFAULT_CHUNK_TOKENS, "default_index_target": DEFAULT_SEARCH_INDEX_TARGET, "indexes": indexes}


@app.get("/api/emails")
def api_list_emails(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    chunk_tokens: int = Query(DEFAULT_CHUNK_TOKENS),
    index_target: str = Query(DEFAULT_SEARCH_INDEX_TARGET),
):
    return {
        "emails": list_emails(
            limit=limit,
            offset=offset,
            db_path=db_path_for_search_target(index_target, chunk_tokens),
        )
    }


@app.get("/api/emails/{email_id}")
def api_get_email(
    email_id: int,
    chunk_tokens: int = Query(DEFAULT_CHUNK_TOKENS),
    index_target: str = Query(DEFAULT_SEARCH_INDEX_TARGET),
):
    return get_email(email_id, db_path=db_path_for_search_target(index_target, chunk_tokens))


@app.post("/api/search")
def api_search(request: SearchRequest):
    plan = extract_search_plan(request.query)
    search_query = " ".join(plan.get("keywords") or [request.query])
    return {
        "plan": plan,
        "results": search_emails(
            search_query,
            limit=max(1, min(request.limit, 100)),
            db_path=db_path_for_search_target(request.index_target, request.chunk_tokens),
            plan=plan,
        ),
    }


@app.post("/api/chat")
def api_chat(request: ChatRequest):
    started = time.time()
    plan = extract_search_plan(request.message)
    search_query = " ".join(plan.get("keywords") or [request.message])
    results = search_emails(
        search_query,
        limit=max(1, min(request.limit, 50)),
        db_path=db_path_for_search_target(request.index_target, request.chunk_tokens),
        plan=plan,
    )
    if not results:
        return {
            "answer": "No indexed emails matched the question. Run `python3 app.py --index --chunk-tokens 512` or `python3 app.py --index --chunk-tokens 256` first.",
            "plan": plan,
            "results": [],
            "elapsed_seconds": round(time.time() - started, 2),
        }
    prompt = build_rag_prompt(request.message, results)
    answer = run_codex_exec(prompt)
    return {
        "answer": answer,
        "plan": plan,
        "results": results,
        "elapsed_seconds": round(time.time() - started, 2),
    }

def main():
    parser = argparse.ArgumentParser(description="Email RAG app")
    parser.add_argument("--index", action="store_true", help="Build or update the SQLite embedding index.")
    parser.add_argument("--reindex", action="store_true", help="Delete and rebuild the full index.")
    parser.add_argument("--email-dir", default=str(DEFAULT_EMAIL_DIR))
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--index-target",
        choices=["chunk", "title", "content", "email", "all"],
        default="chunk",
        help="Index target: chunk=metadata+body chunks, title=subject only, content=body chunks, email=subject+body as one vector, all=all targets.",
    )
    parser.add_argument("--serve", action="store_true", help="Run a local uvicorn server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.index_target == "all" and args.db:
        raise SystemExit("--db cannot be used with --index-target all")

    db_path = Path(args.db) if args.db else db_path_for_index_target(
        "chunk" if args.index_target == "all" else args.index_target,
        args.chunk_tokens,
    )
    set_active_db(db_path)

    if args.index or args.reindex:
        targets = ["chunk", "title", "content", "email"] if args.index_target == "all" else [args.index_target]
        for target in targets:
            build_index(
                Path(args.email_dir),
                Path(args.db) if args.db else db_path_for_index_target(target, args.chunk_tokens),
                reindex=args.reindex,
                chunk_tokens=args.chunk_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
                index_target=target,
            )

    if args.serve:
        import uvicorn
        os.environ["EMAIL_RAG_DEFAULT_CHUNK_TOKENS"] = str(args.chunk_tokens)
        uvicorn.run("app:app", host=args.host, port=args.port, reload=True)

    if not args.index and not args.reindex and not args.serve:
        parser.print_help()


if __name__ == "__main__":
    main()
