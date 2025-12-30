# cpd_app.py

import os
import re
import sys
from pathlib import Path
import importlib.util
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

# =======================  import of cpd_slicer.py =======================
SLICER_PATH = os.environ.get("CPD_SLICER_PATH", str(Path(__file__).parent / "cpd_slicer.py"))

MOD_NAME = "cpd_cg_agent"
if MOD_NAME in sys.modules:
    _mod = sys.modules[MOD_NAME]
else:
    spec = importlib.util.spec_from_file_location(MOD_NAME, SLICER_PATH)
    if spec is None or spec.loader is None:
        st.error(f"Could not load module at {SLICER_PATH}")
        st.stop()
    _mod = importlib.util.module_from_spec(spec)
    sys.modules[MOD_NAME] = _mod
    try:
        spec.loader.exec_module(_mod)  # type: ignore
    except Exception as e:
        st.error(f"Import error loading cpd_slicer.py at {SLICER_PATH}:\n{e}")
        st.stop()

cg_app = getattr(_mod, "app", None)
if cg_app is None:
    st.error("cpd_slicer.py must export `app = graph.compile()`")
    st.stop()

list_testcases = getattr(_mod, "list_testcases", None)
if list_testcases is None:
    st.error("cpd_slicer.py must export `list_testcases()`")
    st.stop()

# ================================ UI helpers =================================
def _extract_tokens_from_messages(messages):
    """
    Backward/forward compatible token extraction:
    - Prefer result['token_usage'] if present
    - Else parse 'TOKENS: prompt=... completion=... total=...' from messages
    """
    if not messages:
        return {}
    rx = re.compile(r"TOKENS:\s*prompt=(\d+|None)\s+completion=(\d+|None)\s+total=(\d+|None)", re.I)
    for m in messages:
        txt = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
        mm = rx.search(txt)
        if mm:
            def _to_int(x):
                return None if x in ("None", None) else int(x)
            return {
                "prompt_tokens": _to_int(mm.group(1)),
                "completion_tokens": _to_int(mm.group(2)),
                "total_tokens": _to_int(mm.group(3)),
            }
    return {}

def _save_upload_to_temp(uploaded, suffix: str):
    if uploaded is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name

# ================================ App state ==================================
if "usage_history" not in st.session_state:
    st.session_state["usage_history"] = []
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

# ================================== Layout ==================================
st.set_page_config(page_title="Dependency Detector", layout="wide")
st.title("Dependency Detector - mini prototype")

with st.sidebar:
    st.header("Run settings")

    model = st.selectbox(
        "Model",
        options=[
            os.environ.get("MODEL", "gpt-4o-mini"),
            "gpt-4.1-mini",
        ],
        index=0
    )

    context_mode = st.selectbox(
        "Context mode",
        options=[
            ("cg_only", "Callgraph only"),
            ("docs_only", "Docs only"),
            ("docs_cg", "Docs + callgraph"),
            ("docs_cg_code", "Docs + callgraph + code"),
        ],
        format_func=lambda x: x[1],
    )[0]

    max_hops = st.slider("Callgraph hop limit (anchors)", min_value=1, max_value=4, value=2, step=1)

    anchor_functions = st.text_input(
        "Anchor function(s) (comma-separated)",
        value="",
        help="Optional but recommended for callgraph modes. Example: ext4_fill_super, ext4_parse_param"
    )

    st.caption(
        ""
        ""
    )

# =============================== Inputs ======================================
colL, colR = st.columns([1, 1])

with colL:
    st.subheader("Artifacts")
    cg_file = st.file_uploader("Upload callgraph (.txt)", type=["txt"])
    doc_file = st.file_uploader("Upload docs/man-page (.txt) (optional)", type=["txt"])
    code_file = st.file_uploader("Upload source code (.c/.txt) (optional)", type=["c", "h", "txt"])

    # if cg_file:
    #     with st.expander("Callgraph preview (first 200 lines)"):
    #         try:
    #             text = cg_file.getvalue().decode("utf-8", errors="ignore")
    #         except Exception:
    #             text = str(cg_file.getvalue()[:5000])
    #         st.code("\n".join(text.splitlines()[:200]))

    # if doc_file:
    #     with st.expander("Docs preview (first 200 lines)"):
    #         dt = doc_file.getvalue().decode("utf-8", errors="ignore")
    #         st.code("\n".join(dt.splitlines()[:200]))

    # if code_file:
    #     with st.expander("Code preview (first 200 lines)"):
    #         ct = code_file.getvalue().decode("utf-8", errors="ignore")
    #         st.code("\n".join(ct.splitlines()[:200]))

with colR:
    st.subheader("Choose one CPD testcase (ground truth)")
    tcs = list_testcases()
    labels = [f"{tc['id']} — {tc['component']}: {tc['param_a']}  vs  {tc['param_b']}" for tc in tcs]
    idx = st.selectbox("Testcase", options=list(range(len(tcs))), format_func=lambda i: labels[i])
    tc = tcs[idx]

    st.markdown("**Ground truth:**")
    st.write(
        {
            "component": tc["component"],
            "A": tc["param_a"],
            "B": tc["param_b"],
            "relation": tc["relation"],
            "direction": tc["direction"],
            "note": tc["note"],
        }
    )

    needs_cg = context_mode in ("cg_only", "docs_cg", "docs_cg_code")
    needs_docs = context_mode in ("docs_only", "docs_cg", "docs_cg_code")
    needs_code = context_mode == "docs_cg_code"

    missing = []
    if needs_cg and cg_file is None:
        missing.append("callgraph")
    if needs_docs and doc_file is None:
        # docs are optional in practice; only warn
        pass
    if needs_code and code_file is None:
        # code is optional in practice; only warn
        pass

    if missing:
        st.warning(f"To run in this mode, you must upload: {', '.join(missing)}")

    go = st.button("Run CPD analysis", disabled=bool(missing))

# =============================== Execute =====================================
if go:
    cg_path = _save_upload_to_temp(cg_file, suffix=".txt") if cg_file else ""
    doc_path = _save_upload_to_temp(doc_file, suffix=".txt") if doc_file else ""
    code_path = _save_upload_to_temp(code_file, suffix=".c") if code_file else ""

    input_state = {
        "cg_path": cg_path,
        "doc_path": doc_path,
        "code_path": code_path,
        "testcase_id": tc["id"],
        "anchor_functions": anchor_functions,
        "context_mode": context_mode,
        "max_hops": int(max_hops),
        "model": model,
    }

    try:
        result = cg_app.invoke(input_state)
        st.session_state["last_result"] = result
    except Exception as e:
        st.error(f"Run failed:\n{e}")
        st.stop()

    # token usage
    token_usage = result.get("token_usage") or _extract_tokens_from_messages(result.get("messages") or [])
    st.session_state["usage_history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "testcase": tc["id"],
        "pred_relation": result.get("pred_relation"),
        "pred_direction": result.get("pred_direction"),
        "match": result.get("match"),
        "prompt_tokens": token_usage.get("prompt_tokens"),
        "completion_tokens": token_usage.get("completion_tokens"),
        "total_tokens": token_usage.get("total_tokens"),
    })

# =============================== Results =====================================
res = st.session_state.get("last_result")
if res:
    st.divider()
    st.subheader("Result")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**LLM prediction**")
        st.write({
            "relation": res.get("pred_relation"),
            "direction": res.get("pred_direction"),
            "confidence": res.get("confidence"),
            "match_ground_truth": bool(res.get("match")),
        })

        if res.get("rationale"):
            with st.expander("Rationale"):
                st.write(res.get("rationale"))

        ev = res.get("evidence") or []
        if ev:
            with st.expander("Evidence bullets (from model)"):
                for b in ev:
                    st.write(f"- {b}")

    with right:
        st.markdown("**Ground truth**")
        st.write({
            "relation": tc["relation"],
            "direction": tc["direction"],
            "note": tc["note"],
        })

        with st.expander("Context used"):
            st.markdown("**Docs excerpt**")
            st.code(res.get("doc_context") or "(empty)")
            st.markdown("**Callgraph context**")
            st.code(res.get("cg_context") or "(empty)")
            st.markdown("**Code context**")
            st.code(res.get("code_context") or "(empty)")

    with st.expander("Raw model output (debug)"):
        st.code(res.get("llm_raw") or "")

# =============================== History =====================================
st.divider()
st.subheader("Run history (tokens + match)")

hist = st.session_state.get("usage_history", [])
if hist:
    df = pd.DataFrame(hist)
    st.dataframe(df, use_container_width=True)
    # quick summary
    try:
        acc = df["match"].mean()
        st.caption(f"Accuracy over runs: {acc:.2%}" if pd.notna(acc) else "")
    except Exception:
        pass
else:
    st.info("No runs yet. Upload artifacts and click 'Run CPD analysis'.")
