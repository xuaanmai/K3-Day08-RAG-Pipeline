"""IELTS Writing Band Descriptor RAG chatbot UI.

Run:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

st.set_page_config(
    page_title="IELTS Writing Lab",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
  --ink:#17221c; --forest:#123d2d; --forest-2:#1d5841;
  --lime:#cef36d; --paper:#faf9f3; --muted:#68746d; --line:#dfe6df;
}
.stApp {
  background: radial-gradient(circle at 88% 4%,rgba(206,243,109,.22),transparent 25rem),
              linear-gradient(180deg,#fbfaf6,#f4f7f3);
  color:var(--ink);
}
html,body,[class*="css"] {font-family:Inter,"Segoe UI",sans-serif}
h1,h2,h3 {letter-spacing:-.035em!important}
[data-testid="stSidebar"] {background:var(--forest);border:0}
[data-testid="stSidebar"] * {color:#f4f7f4}
[data-testid="stSidebar"] .stCaptionContainer {color:#b9c9bf}
[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.12)}
[data-testid="stSidebar"] .stButton button {
  min-height:43px;text-align:left;border-radius:12px;color:#edf4ef;
  background:rgba(255,255,255,.055);border:1px solid rgba(255,255,255,.13)
}
[data-testid="stSidebar"] .stButton button:hover {
  color:var(--ink);background:var(--lime);border-color:var(--lime)
}
.logo {display:grid;place-items:center;width:44px;height:44px;margin-bottom:14px;
  border-radius:14px;background:var(--lime);color:var(--forest);font-size:23px;font-weight:800}
.hero {padding:32px 36px;margin:4px 0 22px;border:1px solid rgba(18,61,45,.1);
  border-radius:25px;background:rgba(255,255,255,.75);box-shadow:0 18px 55px rgba(34,59,45,.07)}
.eyebrow {color:#3f7058;font-size:.76rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}
.hero h1 {margin:8px 0 10px;font-size:clamp(2.15rem,4vw,3.8rem);line-height:1.06;color:var(--ink)}
.hero p {max-width:760px;margin:0;color:var(--muted);font-size:1.04rem;line-height:1.65}
.pills {display:flex;flex-wrap:wrap;gap:8px;margin-top:20px}
.pill {display:inline-flex;align-items:center;gap:7px;padding:7px 11px;border:1px solid var(--line);
  border-radius:99px;background:#fff;color:#425047;font-size:.81rem;font-weight:650}
.dot {width:7px;height:7px;border-radius:50%;background:#53a675}.dot.warn{background:#dfa642}
.criterion {height:100%;padding:16px;border:1px solid var(--line);border-radius:17px;background:rgba(255,255,255,.72)}
.criterion b {color:var(--forest)} .criterion span {display:block;margin-top:5px;color:var(--muted);font-size:.84rem}
[data-testid="stChatMessage"] {padding:1.05rem 1.15rem;margin-bottom:.8rem;border:1px solid var(--line);
  border-radius:18px;background:rgba(255,255,255,.88);box-shadow:0 8px 28px rgba(45,63,52,.045)}
[data-testid="stChatInput"] {border-radius:17px;box-shadow:0 12px 36px rgba(28,52,40,.11)}
[data-testid="stExpander"] {border:1px solid var(--line);border-radius:14px;background:#fff}
.source {padding:11px 13px;margin:8px 0;border-left:3px solid #72a985;border-radius:0 10px 10px 0;background:#f5f8f5}
.source b {color:#263a2e}.source small {color:#718078}
.welcome {padding:23px;border:1px dashed #cbd6cd;border-radius:17px;text-align:center;color:#68756c;background:rgba(255,255,255,.42)}
#MainMenu,footer {visibility:hidden}
</style>
""",
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def _friendly_source_name(value: object, fallback: str) -> str:
    labels = {
        "ielts_writing_band_descriptors": "IELTS Writing Band Descriptors, 2023",
        "ielts_speaking_band_descriptors": "IELTS Speaking Band Descriptors",
        "guide_to_ielts_scores_2025": "Guide to IELTS Scores, 2025",
        "guide-to-ielts-scores-2025": "Guide to IELTS Scores, 2025",
        "general_training_writing_samples": "IELTS General Training Writing Samples",
        "general-training-writing-sample-candidate-responses-and-examiner-comments": (
            "IELTS General Training Writing Samples & Examiner Comments"
        ),
    }
    raw = str(value or "").strip()
    if not raw:
        return fallback
    filename = raw.replace("\\", "/").rsplit("/", 1)[-1]
    stem = filename.rsplit(".", 1)[0]
    if stem.casefold() in labels:
        return labels[stem.casefold()]
    if raw.startswith(("http://", "https://")) or (" " in raw and "_" not in raw):
        return raw
    return stem.replace("_", " ").replace("-", " ").strip().title() or fallback


def _source_info(item: dict, index: int) -> tuple[str, str, float]:

    metadata = item.get("metadata") or {}
    raw_name = (
        metadata.get("title")
        or metadata.get("source")
        or metadata.get("filename")
        or item.get("title")
        or f"Tài liệu {index}"
    )
    name = _friendly_source_name(raw_name, fallback=f"Tài liệu {index}")
    kind = metadata.get("document_type") or metadata.get("type") or "IELTS Writing"
    try:
        score = float(item.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    return str(name), str(kind), score


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Bằng chứng sử dụng · {len(sources)} đoạn"):
        for index, item in enumerate(sources, 1):
            name, kind, score = _source_info(item, index)
            st.markdown(
                f'<div class="source"><b>[{index}] {name}</b><br>'
                f'<small>{kind} · relevance {score:.3f}</small></div>',
                unsafe_allow_html=True,
            )
            excerpt = str(item.get("content", "")).strip()
            if excerpt:
                st.caption(excerpt[:440] + ("…" if len(excerpt) > 440 else ""))


with st.sidebar:
    st.markdown('<div class="logo">✦</div>', unsafe_allow_html=True)
    st.title("IELTS Writing Lab")
    st.caption("Tra cứu tiêu chí chấm điểm và phân tích bài mẫu dựa trên nguồn IELTS chính thức.")

    st.divider()
    st.subheader("Truy vấn mẫu")
    suggestions = [
        "Sự khác biệt giữa Band 6 và Band 7 ở Lexical Resource Task 2 là gì?",
        "Cho ví dụ về cohesive devices đạt Band 8 trong bài Cause and Effect.",
        "Band 8 Task Response cần đáp ứng những điều kiện nào?",
        "Vì sao bài essay mẫu này chưa đạt Band 8?",
    ]
    for i, suggestion in enumerate(suggestions):
        if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
            st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("Thiết lập")
    task_scope = st.selectbox("Phạm vi", ["Task 2", "Task 1", "Cả hai"])
    top_k = st.slider("Số đoạn bằng chứng", 3, 10, 5)
    debug = st.toggle("Hiện thông tin kỹ thuật", False)
    if st.button("Xoá cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    st.divider()
    st.caption("Band Descriptors · Essay Samples · Examiner Comments")


has_key = bool(
    os.getenv("GEMINI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
key_class = "" if has_key else "warn"
key_text = "LLM đã cấu hình" if has_key else "Chưa có LLM API key"

st.markdown(
    f"""
<section class="hero">
  <div class="eyebrow">IELTS Writing · Evidence-first assistant</div>
  <h1>Hiểu band score.<br>Viết có chiến lược.</h1>
  <p>Đối chiếu tiêu chí Band 6–9, giải thích điểm mạnh và điểm giới hạn của essay mẫu,
  đồng thời dẫn lại đúng bằng chứng từ band descriptors và nhận xét examiner.</p>
  <div class="pills">
    <span class="pill"><span class="dot"></span>Official descriptors</span>
    <span class="pill"><span class="dot"></span>Hybrid RAG</span>
    <span class="pill"><span class="dot {key_class}"></span>{key_text}</span>
  </div>
</section>
""",
    unsafe_allow_html=True,
)

cols = st.columns(4)
criteria = [
    ("Task Response", "Trả lời đúng, đủ và phát triển lập luận"),
    ("Coherence & Cohesion", "Tổ chức ý, đoạn văn và liên kết"),
    ("Lexical Resource", "Độ rộng, chính xác và tự nhiên của từ vựng"),
    ("Grammar", "Độ đa dạng và chính xác của cấu trúc"),
]
for col, (name, description) in zip(cols, criteria):
    with col:
        st.markdown(
            f'<div class="criterion"><b>{name}</b><span>{description}</span></div>',
            unsafe_allow_html=True,
        )

st.write("")
if not st.session_state.messages:
    st.markdown(
        '<div class="welcome">Chọn câu hỏi mẫu hoặc nhập một câu hỏi về IELTS Writing bên dưới.</div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))
            if debug and message.get("retrieval_source"):
                st.caption(f"Retrieval route: `{message['retrieval_source']}`")

typed_query = st.chat_input("Hỏi về Band 6–9, bốn tiêu chí chấm điểm hoặc essay mẫu…")
query = typed_query or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    scoped_query = f"[Phạm vi IELTS Writing {task_scope}] {query}"
    conversation_history = [
        {"role": message["role"], "content": message["content"]}
        for message in st.session_state.messages[-6:]
    ]
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("Đang đối chiếu band descriptors…", expanded=False) as status:
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    scoped_query,
                    top_k=top_k,
                    conversation_history=conversation_history,
                )
                answer = response.get("answer") or "Không tìm thấy đủ bằng chứng để trả lời."
                sources = response.get("sources") or []
                route = response.get("retrieval_source", "unknown")
                status.update(label="Đã tổng hợp câu trả lời", state="complete")
            except NotImplementedError:
                answer = (
                    "Giao diện đã sẵn sàng nhưng pipeline sinh câu trả lời trong "
                    "`src/task9_retrieval_pipeline.py` và `src/task10_generation.py` chưa hoàn thiện."
                )
                sources, route = [], "not_implemented"
                status.update(label="Pipeline RAG chưa hoàn thiện", state="error")
            except Exception as exc:
                answer = "Không thể chạy pipeline. Hãy kiểm tra ChromaDB, embedding model và API key trong `.env`."
                sources, route = [], "error"
                status.update(label="Có lỗi khi chạy pipeline", state="error")
                if debug:
                    st.exception(exc)

        st.markdown(answer)
        render_sources(sources)
        if debug:
            st.caption(f"Retrieval route: `{route}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "retrieval_source": route}
    )
