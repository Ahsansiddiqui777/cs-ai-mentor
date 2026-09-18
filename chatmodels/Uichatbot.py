import os
import io
import base64
import sqlite3
import uuid
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

load_dotenv()

st.set_page_config(page_title="Start Your CS Journey", page_icon="💻", layout="wide")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.db")

SYSTEM_PROMPT = (
    "You are an expert Computer Science professor and AI mentor named 'Start Your CS Journey', "
    "created by Ahsan Siddiqui. Your goal is to guide students from absolute beginner concepts "
    "(Variables, Loops, Functions) up to advanced levels (Data Structures, Algorithms, System Design, "
    "Machine Learning, and Generative AI). Always provide clear explanations, code examples where "
    "applicable, practice questions, and step-by-step guidance. If a student attaches a file (document "
    "text or an image), use its contents to answer their question.\n\n"
    "If the student asks who created you, who Ahsan Siddiqui is, or how to contact/follow the creator, "
    "mention that Ahsan Siddiqui's LinkedIn is https://www.linkedin.com/in/ahsan-siddiqui-37320a2b5/ "
    "and encourage them to connect or follow him there. Don't bring this up unless the student asks "
    "about the creator."
)

TEXT_EXTENSIONS = {"txt", "md", "csv", "py", "json"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_EXTRACT_CHARS = 8000


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

            h1, h2, h3 { font-family: 'Poppins', sans-serif !important; }

            .hero-title {
                background: linear-gradient(90deg, #4F46E5, #06B6D4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: 'Poppins', sans-serif;
                font-weight: 700;
                font-size: 2.4rem;
                margin-bottom: 0;
            }
            .hero-subtitle {
                color: #6B7280;
                font-size: 1.05rem;
                margin-top: 0.2rem;
                margin-bottom: 1rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
                border-right: 1px solid #E5E7EB;
            }

            div[data-testid="stChatMessage"] {
                border-radius: 16px;
                padding: 0.6rem 0.9rem;
                margin-bottom: 0.6rem;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06);
            }

            .stButton>button {
                border-radius: 10px;
                font-weight: 500;
                transition: all 0.15s ease-in-out;
            }
            .stButton>button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 10px rgba(79, 70, 229, 0.15);
            }

            div[data-testid="stChatInput"] textarea {
                border-radius: 14px !important;
            }

            .attach-chip {
                display: inline-block;
                background: #EEF2FF;
                color: #4338CA;
                border-radius: 999px;
                padding: 0.25rem 0.8rem;
                font-size: 0.82rem;
                font-weight: 500;
                margin-bottom: 0.6rem;
            }

            .starter-card {
                border: 1px solid #E5E7EB;
                border-radius: 12px;
                padding: 0.9rem;
                background: #FAFAFF;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Database layer
# ----------------------------------------------------------------------------
@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chats (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL,
            title       TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
        """
    )
    # Migrate older DBs created before multi-user support existed
    chat_cols = {row[1] for row in conn.execute("PRAGMA table_info(chats)")}
    if "user_id" not in chat_cols:
        conn.execute("ALTER TABLE chats ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy'")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id     TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            img_data    TEXT,
            img_mime    TEXT,
            file_name   TEXT,
            created_at  TEXT NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
        """
    )
    # Migrate older DBs created before attachment columns existed
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    for col in ("img_data", "img_mime", "file_name"):
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE messages ADD COLUMN {col} TEXT")
    conn.commit()
    return conn


def now():
    return datetime.now().isoformat(timespec="seconds")


def create_chat(user_id, title="New chat"):
    conn = get_conn()
    chat_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO chats (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
        (chat_id, user_id, title, now(), now()),
    )
    conn.commit()
    return chat_id


def list_chats(user_id, search=""):
    conn = get_conn()
    if search:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM chats WHERE user_id = ? AND title LIKE ? "
            "ORDER BY updated_at DESC",
            (user_id, f"%{search}%"),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, title, updated_at FROM chats WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return rows


def rename_chat(chat_id, new_title):
    conn = get_conn()
    conn.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title.strip(), chat_id))
    conn.commit()


def delete_chat(chat_id, user_id):
    conn = get_conn()
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id))
    conn.commit()


def delete_all_chats(user_id):
    conn = get_conn()
    ids = [r[0] for r in conn.execute("SELECT id FROM chats WHERE user_id = ?", (user_id,))]
    for cid in ids:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (cid,))
    conn.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
    conn.commit()


def add_message(chat_id, role, content, img_data=None, img_mime=None, file_name=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content, img_data, img_mime, file_name, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (chat_id, role, content, img_data, img_mime, file_name, now()),
    )
    conn.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now(), chat_id))
    conn.commit()


def get_messages(chat_id):
    conn = get_conn()
    return conn.execute(
        "SELECT role, content, img_data, img_mime, file_name FROM messages "
        "WHERE chat_id = ? ORDER BY id ASC",
        (chat_id,),
    ).fetchall()


def delete_last_exchange(chat_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, role FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT 1",
        (chat_id,),
    ).fetchone()
    if row and row[1] == "assistant":
        conn.execute("DELETE FROM messages WHERE id = ?", (row[0],))
        conn.commit()


# ----------------------------------------------------------------------------
# File handling
# ----------------------------------------------------------------------------
def extract_text_from_pdf(raw_bytes):
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[Could not read PDF: the 'pypdf' package is not installed. Run: pip install pypdf]"
    reader = PdfReader(io.BytesIO(raw_bytes))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text).strip()


def extract_text_from_docx(raw_bytes):
    try:
        import docx
    except ImportError:
        return "[Could not read DOCX: the 'python-docx' package is not installed. Run: pip install python-docx]"
    document = docx.Document(io.BytesIO(raw_bytes))
    return "\n".join(p.text for p in document.paragraphs).strip()


def process_upload(uploaded_file):
    """Returns a dict: {kind: 'text'|'image', name, content/(data,mime)}"""
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    raw = uploaded_file.getvalue()

    if ext in IMAGE_EXTENSIONS:
        b64 = base64.b64encode(raw).decode("utf-8")
        mime = f"image/{'jpeg' if ext == 'jpg' else ext}"
        return {"kind": "image", "name": uploaded_file.name, "data": b64, "mime": mime}

    if ext == "pdf":
        text = extract_text_from_pdf(raw)
    elif ext == "docx":
        text = extract_text_from_docx(raw)
    elif ext in TEXT_EXTENSIONS:
        text = raw.decode("utf-8", errors="ignore")
    else:
        text = f"[Unsupported file type: .{ext}]"

    text = text[:MAX_EXTRACT_CHARS]
    return {"kind": "text", "name": uploaded_file.name, "content": text}


# ----------------------------------------------------------------------------
# Model layer
# ----------------------------------------------------------------------------
@st.cache_resource
def get_model(model_name, temperature):
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)


def to_langchain(rows):
    msgs = [SystemMessage(content=SYSTEM_PROMPT)]
    for role, content, img_data, img_mime, file_name in rows:
        if role == "user":
            if img_data:
                msgs.append(
                    HumanMessage(
                        content=[
                            {"type": "text", "text": content},
                            {"type": "image_url", "image_url": f"data:{img_mime};base64,{img_data}"},
                        ]
                    )
                )
            else:
                msgs.append(HumanMessage(content=content))
        else:
            msgs.append(AIMessage(content=content))
    return msgs


def chunk_text(stream):
    for chunk in stream:
        c = chunk.content
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            for item in c:
                if isinstance(item, dict) and "text" in item:
                    yield item["text"]
                elif isinstance(item, str):
                    yield item


def make_title(text):
    text = " ".join(text.split())
    return text[:45] + ("…" if len(text) > 45 else "")


def export_markdown(chat_id, title):
    lines = [f"# {title}", ""]
    for role, content, img_data, img_mime, file_name in get_messages(chat_id):
        lines.append(f"**{'You' if role == 'user' else 'Mentor'}:**")
        if file_name:
            lines.append(f"*(attached: {file_name})*")
        lines.append("")
        lines.append(content)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Login gate
# ----------------------------------------------------------------------------
inject_css()

if not st.user.is_logged_in:
    st.markdown('<p class="hero-title">🚀 Start Your CS Journey</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Master Computer Science & AI from Basics to Advanced Level</p>',
        unsafe_allow_html=True,
    )
    st.write("Sign in to save your chats and pick up where you left off, on any device.")
    if st.button("🔐  Continue with Google", type="primary"):
        st.login("google")
    st.stop()

USER_ID = st.user.email

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "chat_id" not in st.session_state:
    chats = list_chats(USER_ID)
    st.session_state.chat_id = chats[0][0] if chats else create_chat(USER_ID)

if "renaming" not in st.session_state:
    st.session_state.renaming = None

if "confirm_delete_all" not in st.session_state:
    st.session_state.confirm_delete_all = False

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("💻 CS & AI Mentor")
    st.caption("Created by **Ahsan Siddiqui**")

    user_name = st.user.name or st.user.email
    st.markdown(f"👋 **{user_name}**")
    if st.button("🚪 Log out", use_container_width=True):
        st.logout()

    st.markdown("---")

    if st.button("➕  New chat", use_container_width=True, type="primary"):
        st.session_state.chat_id = create_chat(USER_ID)
        st.rerun()

    search = st.text_input("Search chats", placeholder="🔍 Search…", label_visibility="collapsed")

    st.markdown("##### Your chats")
    chats = list_chats(USER_ID, search)

    if not chats:
        st.caption("No chats yet.")

    for chat_id, title, updated in chats:
        is_active = chat_id == st.session_state.chat_id

        if st.session_state.renaming == chat_id:
            new_title = st.text_input(
                "Rename", value=title, key=f"rename_{chat_id}", label_visibility="collapsed"
            )
            c1, c2 = st.columns(2)
            if c1.button("Save", key=f"save_{chat_id}", use_container_width=True):
                if new_title.strip():
                    rename_chat(chat_id, new_title)
                st.session_state.renaming = None
                st.rerun()
            if c2.button("Cancel", key=f"cancel_{chat_id}", use_container_width=True):
                st.session_state.renaming = None
                st.rerun()
        else:
            col1, col2, col3 = st.columns([7, 1, 1])
            label = ("🟢 " if is_active else "") + title
            if col1.button(label, key=f"open_{chat_id}", use_container_width=True):
                st.session_state.chat_id = chat_id
                st.rerun()
            if col2.button("✏️", key=f"edit_{chat_id}", help="Rename"):
                st.session_state.renaming = chat_id
                st.rerun()
            if col3.button("🗑️", key=f"del_{chat_id}", help="Delete"):
                delete_chat(chat_id, USER_ID)
                if st.session_state.chat_id == chat_id:
                    remaining = list_chats(USER_ID)
                    st.session_state.chat_id = remaining[0][0] if remaining else create_chat(USER_ID)
                st.rerun()

    st.markdown("---")

    with st.expander("📚  Topics"):
        st.markdown(
            """
            * 🟢 **Basics:** Python, Logic, Variables, Loops
            * 🟡 **Core CS:** Data Structures, Algorithms, OOP
            * 🔴 **Advanced:** System Design, Operating Systems
            * 🤖 **AI Field:** Machine Learning, Neural Networks, GenAI
            """
        )

    current_title = next(
        (t for i, t, _ in list_chats(USER_ID) if i == st.session_state.chat_id), "chat"
    )
    st.download_button(
        "⬇️  Export this chat",
        data=export_markdown(st.session_state.chat_id, current_title),
        file_name=f"{current_title[:30].replace(' ', '_')}.md",
        mime="text/markdown",
        use_container_width=True,
    )

    if not st.session_state.confirm_delete_all:
        if st.button("🧹  Delete all chats", use_container_width=True):
            st.session_state.confirm_delete_all = True
            st.rerun()
    else:
        st.warning("Delete every chat permanently?")
        c1, c2 = st.columns(2)
        if c1.button("Yes, delete", use_container_width=True):
            delete_all_chats(USER_ID)
            st.session_state.chat_id = create_chat(USER_ID)
            st.session_state.confirm_delete_all = False
            st.rerun()
        if c2.button("Cancel", use_container_width=True):
            st.session_state.confirm_delete_all = False
            st.rerun()


# ----------------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------------
st.markdown('<p class="hero-title">🚀 Start Your CS Journey</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Master Computer Science & AI from Basics to Advanced Level</p>',
    unsafe_allow_html=True,
)

model = get_model("gemini-3.6-flash", 0.3)
history = get_messages(st.session_state.chat_id)

if not history:
    st.info("Pick a starting point, attach a file, or just ask anything below.")
    cols = st.columns(3)
    starters = [
        "Teach me Python loops from scratch",
        "Explain Big-O notation with examples",
        "What is a neural network, simply?",
    ]
    for col, starter in zip(cols, starters):
        with col:
            st.markdown('<div class="starter-card">', unsafe_allow_html=True)
            if st.button(starter, use_container_width=True):
                st.session_state.pending = starter
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

for role, content, img_data, img_mime, file_name in history:
    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🤖"):
        if file_name:
            st.markdown(f'<span class="attach-chip">📎 {file_name}</span>', unsafe_allow_html=True)
        if img_data:
            st.image(base64.b64decode(img_data))
        st.markdown(content)

# --- Attachment + input row ---
with st.expander("📎 Attach a file (PDF, DOCX, TXT, or image)", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "docx", "txt", "md", "csv", "png", "jpg", "jpeg", "webp"],
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.caption(f"Ready to send with your next message: **{uploaded_file.name}**")

prompt = st.chat_input("Ask a topic (e.g., 'Teach me Python loops')…")

if "pending" in st.session_state:
    prompt = st.session_state.pop("pending")

if prompt:
    img_data = img_mime = file_name = None
    display_prompt = prompt

    if uploaded_file:
        processed = process_upload(uploaded_file)
        file_name = processed["name"]
        if processed["kind"] == "image":
            img_data, img_mime = processed["data"], processed["mime"]
        else:
            display_prompt = (
                f"{prompt}\n\n--- Attached file: {file_name} ---\n{processed['content']}"
            )
        st.session_state.uploader_key += 1  # reset the uploader widget

    add_message(st.session_state.chat_id, "user", display_prompt, img_data, img_mime, file_name)

    if len(history) == 0:
        rename_chat(st.session_state.chat_id, make_title(prompt))

    with st.chat_message("user", avatar="🧑‍💻"):
        if file_name:
            st.markdown(f'<span class="attach-chip">📎 {file_name}</span>', unsafe_allow_html=True)
        if img_data:
            st.image(base64.b64decode(img_data))
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            lc_messages = to_langchain(get_messages(st.session_state.chat_id))
            reply = st.write_stream(chunk_text(model.stream(lc_messages)))
            add_message(st.session_state.chat_id, "assistant", reply)
        except Exception as e:
            st.error(f"API Error: {e}")

    st.rerun()

if history:
    c1, _ = st.columns([1, 5])
    if c1.button("🔄 Regenerate"):
        delete_last_exchange(st.session_state.chat_id)
        rows = get_messages(st.session_state.chat_id)
        if rows and rows[-1][0] == "user":
            try:
                reply = "".join(chunk_text(model.stream(to_langchain(rows))))
                add_message(st.session_state.chat_id, "assistant", reply)
            except Exception as e:
                st.error(f"API Error: {e}")
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)
st.caption("This platform is designed by **Ahsan Siddiqui** to help students master CS and AI fields.")
