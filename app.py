import os, re, sys
from pathlib import Path
import importlib.util
import tempfile
import difflib
import hashlib
from datetime import datetime

import pandas as pd
import streamlit as st

# ======================= Dynamic import of slicer.py =======================
SLICER_PATH = os.environ.get("SLICER_PATH", str(Path(__file__).parent / "slicer.py"))

MOD_NAME = "cg_agent"
if MOD_NAME in sys.modules:
    del sys.modules[MOD_NAME]

_spec = importlib.util.spec_from_file_location(MOD_NAME, SLICER_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader, f"Cannot import slicer.py at {SLICER_PATH}"
sys.modules[MOD_NAME] = _mod
try:
    _spec.loader.exec_module(_mod)
except Exception as e:
    st.error(f"Import error loading slicer.py at {SLICER_PATH}:\n{e}")
    st.stop()

cg_app = getattr(_mod, "app", None)
if cg_app is None:
    st.error("slicer.py must export `app = graph.compile()`")
    st.stop()

# Optional helpers from slicer.py
extract_all_functions = getattr(_mod, "extract_all_functions", None)
if extract_all_functions is None:
    _NODE_RX = re.compile(r"^\s*Call graph node for function:\s*'([^']+)'", re.M)
    def extract_all_functions(text: str):
        names = {m.group(1) for m in _NODE_RX.finditer(text)}
        names.discard("<null>")
        return sorted(names, key=str.lower)

# ============================ Local helpers ==============================

def get_file_hash(file_content: bytes) -> str:
    """Generate a hash for file content to detect changes."""
    return hashlib.md5(file_content).hexdigest()


def _remove_function_from_c_source(src_text: str, func: str, debug: bool = False):
    """
    Robustly remove a C function definition named `func` from `src_text`.
    Handles storage qualifiers, attributes, comments, newlines, and braces inside
    strings/comments. Returns (new_text, removed: bool, info_msg).
    """
    text = src_text.replace("\r\n", "\n")

    # Try regex-based approach first for better multi-line signature handling
    pattern = re.compile(
        r'('
        r'(?:^|\n)'
        r'(?:[^\n]*?)'
        r'\b' + re.escape(func) + r'\s*'
        r'\([^)]*\)'
        r'[^{;]*?'
        r'\{'
        r')',
        re.MULTILINE | re.DOTALL
    )

    matches = list(pattern.finditer(text))

    if debug and matches:
        st.info(f"Regex found {len(matches)} potential matches for '{func}'")

    if not matches:
        # Fallback: simple search
        name_idx = text.find(func + "(")
        if name_idx == -1:
            return src_text, False, f"Function '{func}' not found (name search failed)."

        if debug:
            st.warning(f"Using fallback method for '{func}'")

        # Check for definition vs prototype
        semi_idx = text.find(";", name_idx)
        brace_idx = text.find("{", name_idx)

        if brace_idx == -1:
            return src_text, False, f"No opening brace found for '{func}'."

        if semi_idx != -1 and semi_idx < brace_idx:
            # This might be a prototype, try next occurrence
            name_idx = text.find(func + "(", name_idx + 1)
            if name_idx == -1:
                return src_text, False, f"Only prototype found for '{func}'."
            brace_idx = text.find("{", name_idx)
            if brace_idx == -1:
                return src_text, False, f"No opening brace found for '{func}'."

        # Find signature start (go back to capture return type)
        sig_start = text.rfind("\n", 0, name_idx) + 1

        # Check previous lines for return type
        check_pos = sig_start - 1
        while check_pos > 0:
            prev_line_start = text.rfind("\n", 0, check_pos - 1) + 1
            prev_line = text[prev_line_start:check_pos].strip()

            if prev_line and not prev_line.startswith(('//', '/*')) and not prev_line.endswith((';', '}', '{')):
                sig_start = prev_line_start
                check_pos = prev_line_start
            else:
                break

    else:
        # Use regex match
        match = matches[0]
        sig_start = match.start()
        brace_idx = match.end() - 1

        if debug:
            preview = text[sig_start:min(sig_start + 150, len(text))]
            st.code(f"Found at pos {sig_start}:\n{preview}", language="c")

    # Find matching closing brace with proper string/comment handling
    i = brace_idx
    depth = 0
    n = len(text)

    in_string = False
    in_char = False
    in_sl_comment = False
    in_ml_comment = False
    escape_next = False

    while i < n:
        ch = text[i]
        next_ch = text[i + 1] if i + 1 < n else ''

        if escape_next:
            escape_next = False
            i += 1
            continue

        if in_string:
            if ch == '\\':
                escape_next = True
            elif ch == '"':
                in_string = False
        elif in_char:
            if ch == '\\':
                escape_next = True
            elif ch == "'":
                in_char = False
        elif in_sl_comment:
            if ch == '\n':
                in_sl_comment = False
        elif in_ml_comment:
            if ch == '*' and next_ch == '/':
                in_ml_comment = False
                i += 1
        else:
            if ch == '/' and next_ch == '/':
                in_sl_comment = True
                i += 1
            elif ch == '/' and next_ch == '*':
                in_ml_comment = True
                i += 1
            elif ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    body_end = i + 1

                    # Clean up: remove preceding newline if exists
                    cut_start = sig_start
                    if cut_start > 0 and text[cut_start - 1] == '\n':
                        cut_start -= 1

                    # Clean up: remove trailing newline if exists
                    cut_end = body_end
                    if cut_end < n and text[cut_end] == '\n':
                        cut_end += 1

                    new_text = text[:cut_start] + text[cut_end:]
                    chars_removed = cut_end - cut_start

                    if debug:
                        st.success(f"Successfully removed {chars_removed} characters")

                    return new_text, True, f"Removed '{func}' ({chars_removed} chars)."

        i += 1

    return src_text, False, f"Unbalanced braces for '{func}' (final depth={depth})."


def _diff_preview(old: str, new: str, filename="source.c"):
    diff = difflib.unified_diff(
        old.splitlines(True),
        new.splitlines(True),
        fromfile=filename + " (original)",
        tofile=filename + " (updated)",
        n=3
    )
    return "".join(diff)


def _clear_removal_state():
    for k in ("removed_func", "updated_src", "orig_src", "src_name", "removal_info", "saved_file_path"):
        st.session_state.pop(k, None)


def _clear_analysis_state():
    """Clear analysis results but keep uploaded files."""
    for k in ("last_agent_result", "static_decision", "llm_decision", "llm_rationale",
              "direct_callers", "trans_callers", "direct_callees", "trans_callees",
              "snippet", "edge_warnings", "pointer_usage", "func_metadata"):
        st.session_state.pop(k, None)


# ============================== UI LAYOUT ================================

st.set_page_config(page_title="LLM-assisted Code Optimizer using LangGraph ", layout="wide")
st.title("LLM-assisted Code Optimizer using LangGraph")

# Initialize session usage history (persists per session)
if "usage_history" not in st.session_state:
    st.session_state["usage_history"] = []  # list[dict] of per-call usage rows

# --- Render success + download ASAP on each rerun (before other logic) ---
if "updated_src" in st.session_state and "removed_func" in st.session_state:
    target_ok = st.session_state.get("removed_func")
    new_text = st.session_state.get("updated_src", "")
    old_text = st.session_state.get("orig_src", "")
    fname = st.session_state.get("src_name", "source.c")
    info = st.session_state.get("removal_info", "")

    st.success(f"Function **{target_ok}** removed successfully. {info}")

    with st.expander("Diff preview"):
        try:
            st.code(_diff_preview(old_text, new_text, filename=fname), language="diff")
        except Exception as e:
            st.warning(f"Could not render diff: {e}")

    saved_path = st.session_state.get("saved_file_path")
    if saved_path and Path(saved_path).exists():
        st.caption(f"Saved a copy to: `{saved_path}`")
        with open(saved_path, "rb") as fh:
            data_bytes = fh.read()
        st.download_button(
            "Download modified file",
            data=data_bytes,
            file_name=Path(saved_path).name,
            mime="text/x-csrc",
            key=f"download_{target_ok}",
        )
    else:
        base, dot, ext = fname.rpartition(".")
        out_name = base + f"_without_{target_ok}" + (dot + ext if dot else ".c")
        st.download_button(
            "Download modified file",
            data=new_text,
            file_name=out_name,
            mime="text/x-csrc",
            key=f"download_{target_ok}",
        )

    st.divider()

# Sidebar: model selection
with st.sidebar:
    st.subheader("Model")
    default_model = os.environ.get("MODEL", "gpt-3.5-turbo")
    model = st.text_input("LLM model", value=default_model, help="Model name passed to the OpenAI client in slicer.py")

    st.divider()

    # Add reset button
    if st.button("🗑️ Clear all data", help="Clear all uploaded files and analysis results"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# --- Upload callgraph.txt ---
cg_file = st.file_uploader("Upload callgraph", type=["txt"], key="cg")
cg_temp_path, cg_functions = None, []

if cg_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="wb") as tmp:
        tmp.write(cg_file.read())
        cg_temp_path = tmp.name
    st.success(f"Saved callgraph: `{cg_temp_path}`")

    cg_text = Path(cg_temp_path).read_text(encoding="utf-8", errors="ignore")
    with st.expander("Callgraph preview (first ~1200 chars)"):
        st.code(cg_text[:1200], language="text")

    try:
        cg_functions = extract_all_functions(cg_text)
    except Exception as e:
        st.error(f"Failed to parse function names: {e}")

# --- Upload C source (sqlite3.c) with session state mgmt ---
src_file = st.file_uploader("Upload C source", type=["c", "txt"], key="src")

# Initialize session state for source if not exists
if "__src_text__" not in st.session_state:
    st.session_state["__src_text__"] = None
if "__src_name__" not in st.session_state:
    st.session_state["__src_name__"] = "source.c"
if "__src_hash__" not in st.session_state:
    st.session_state["__src_hash__"] = None
if "__src_temp_path__" not in st.session_state:
    st.session_state["__src_temp_path__"] = None

src_temp_path = None
src_text = None

if src_file:
    # Read file content
    file_bytes = src_file.read()
    file_hash = get_file_hash(file_bytes)

    # Check if this is a new/different file
    if file_hash != st.session_state["__src_hash__"]:
        # New file uploaded - process it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".c", mode="wb") as tmp:
            tmp.write(file_bytes)
            src_temp_path = tmp.name

        src_text = file_bytes.decode("utf-8", errors="ignore")

        # Store in session state IMMEDIATELY
        st.session_state["__src_text__"] = src_text
        st.session_state["__src_name__"] = getattr(src_file, "name", "source.c")
        st.session_state["__src_hash__"] = file_hash
        st.session_state["__src_temp_path__"] = src_temp_path

        # Clear any previous removal/analysis state when new file uploaded
        _clear_removal_state()
        _clear_analysis_state()

        st.success(f"Saved C source: `{src_temp_path}` ({len(src_text):,} chars)")
    else:
        # Same file as before - use cached data
        src_text = st.session_state["__src_text__"]
        src_temp_path = st.session_state.get("__src_temp_path__")

        # If temp path doesn't exist, recreate it
        if not src_temp_path or not Path(src_temp_path).exists():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".c", mode="wb") as tmp:
                tmp.write(file_bytes)
                src_temp_path = tmp.name
            st.session_state["__src_temp_path__"] = src_temp_path

    if src_text:
        with st.expander("Source preview (first ~1200 chars)"):
            st.code(src_text[:1200], language="c")
else:
    # No file uploaded currently - check if we have cached data
    if st.session_state["__src_text__"]:
        src_text = st.session_state["__src_text__"]
        src_temp_path = st.session_state.get("__src_temp_path__")
        st.info(f"Using previously uploaded source: {st.session_state['__src_name__']} ({len(src_text):,} chars)")

# ---- Target selector (persistent) ----
if "selected_target" not in st.session_state:
    default_target = "computeJD" if (cg_functions and "computeJD" in cg_functions) else (cg_functions[0] if cg_functions else "")
    st.session_state["selected_target"] = default_target

widget_options = cg_functions or ([st.session_state["selected_target"]] if st.session_state["selected_target"] else [])
selected = st.selectbox("Target function", options=widget_options, key="target_select")

if selected and selected != st.session_state["selected_target"]:
    st.session_state["selected_target"] = selected

target = st.session_state["selected_target"]

# Analyze button requires both files and a target
go = st.button("Analyze", disabled=not (cg_temp_path and src_temp_path and target))

# =========================== Run analyzer ===============================

if go and cg_temp_path and src_temp_path and target:
    _clear_removal_state()  # fresh run
    _clear_analysis_state()  # clear old analysis

    with st.spinner("Analyzing (static + LLM)…"):
        result = cg_app.invoke({
            "cg_path": cg_temp_path,
            "c_path": src_temp_path,
            "target": target,
            "model": model,
            "messages": [],
        })

    # --- Token usage capture + logging ---
    usage = result.get("token_usage") or {}
    # Fallback parse from messages if needed (handles SDK differences)
    if not usage:
        import re as _re
        _RX = _re.compile(r"Usage:\s*prompt=(\d+)\s+completion=(\d+)\s+total=(\d+)")
        for m in (result.get("messages") or []):
            text = (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
            mm = _RX.search(text)
            if mm:
                usage = {"prompt_tokens": int(mm.group(1)), "completion_tokens": int(mm.group(2)), "total_tokens": int(mm.group(3))}
                break

    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        try:
            total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
        except Exception:
            total_tokens = None

    st.session_state["token_usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    # Append usage row to history
    st.session_state["usage_history"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "target": target,
        "llm_decision": result.get("llm_decision"),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })

    # Store other results in session state
    st.session_state["last_agent_result"] = result
    st.session_state["static_decision"] = result.get("decision", "UNKNOWN")
    st.session_state["llm_decision"] = result.get("llm_decision", "DO_NOT_REMOVE")
    st.session_state["llm_rationale"] = result.get("llm_rationale", "")
    st.session_state["direct_callers"] = result.get("callers", [])
    st.session_state["trans_callers"] = result.get("callers_transitive", [])
    st.session_state["direct_callees"] = result.get("callees", [])
    st.session_state["trans_callees"] = result.get("callees_transitive", [])
    st.session_state["snippet"] = result.get("evidence_snippet", "")
    st.session_state["edge_warnings"] = result.get("edge_case_warnings", [])
    st.session_state["pointer_usage"] = result.get("pointer_usage", {})
    st.session_state["func_metadata"] = result.get("function_metadata")

# Display results if we have them (handles reruns from checkbox clicks, etc.)
if "last_agent_result" in st.session_state:
    # Load from session state
    static_decision = st.session_state.get("static_decision", "UNKNOWN")
    llm_decision = st.session_state.get("llm_decision", "DO_NOT_REMOVE")
    llm_rationale = st.session_state.get("llm_rationale", "")
    direct_callers = st.session_state.get("direct_callers", [])
    trans_callers = st.session_state.get("trans_callers", [])
    direct_callees = st.session_state.get("direct_callees", [])
    trans_callees = st.session_state.get("trans_callees", [])
    snippet = st.session_state.get("snippet", "")
    edge_warnings = st.session_state.get("edge_warnings", [])
    pointer_usage = st.session_state.get("pointer_usage", {})
    func_metadata = st.session_state.get("func_metadata")

    st.subheader("Decisions")

    st.markdown(f"**LLM decision:** {('✅ REMOVE' if llm_decision == 'REMOVE' else '⛔ DO_NOT_REMOVE')}")
    if llm_rationale:
        with st.expander("LLM rationale"):
            st.write(llm_rationale)
    st.markdown(
        f"**Static graph decision:** {('✅ SAFE_TO_REMOVE' if static_decision == 'SAFE_TO_REMOVE' else '⚠️ DEPENDENT')}\n\n"
    )
    # Display edge case warnings if any
    if edge_warnings:
        st.subheader("⚠️ Edge Case Analysis")
        for warning in edge_warnings:
            if "CRITICAL" in warning:
                st.error(warning)
            else:
                st.warning(warning)

        # Show additional metadata if available
        if func_metadata:
            with st.expander("Function Metadata"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Call Graph #uses", func_metadata.uses_count if hasattr(func_metadata, 'uses_count') else 0)
                with col2:
                    has_addr = pointer_usage.get("has_address_taken", False)
                    st.metric("Address Taken", "Yes" if has_addr else "No")
                with col3:
                    is_reg = bool(pointer_usage.get("function_registration", []))
                    st.metric("Registered", "Yes" if is_reg else "No")

                # Show context snippets if available
                if pointer_usage.get("context_snippets"):
                    st.write("**Usage Context:**")
                    for i, ctx in enumerate(pointer_usage["context_snippets"][:3], 1):
                        st.code(ctx[:200], language="c")

    with st.expander(" Call Graph Analysis"):
        st.subheader("⬇️ INCOMING: Who calls this function")
        st.markdown(
            f"**Direct callers ({len(direct_callers)}):** "
            + (", ".join(direct_callers) if direct_callers else "*(none)*")
        )

        if not direct_callers:
            st.info("✓ No functions call this function directly")

        st.divider()

        st.subheader("⬆️ OUTGOING: What this function calls")
        st.markdown(
            f"**Direct callees ({len(direct_callees)}):** "
            + (", ".join(direct_callees) if direct_callees else "*(none)*")
        )

        if not direct_callees:
            st.info("✓ This function doesn't call any other functions (leaf node)")

        # Show interpretation
        st.divider()
        st.subheader(" Interpretation")

        if not direct_callers:
            if not direct_callees:
                st.success("**Isolated function**: No incoming or outgoing calls. Likely dead code if no edge cases detected.")
            else:
                st.warning(f"**Dead code with dependencies**: Not called by anyone, but calls {len(direct_callees)} function(s). "
                           f"Removing it could clean up this unused code path (if no edge cases like callbacks/registrations).")
        else:
            st.info(f"**Active function**: Called by {len(direct_callers)} function(s) and calls {len(direct_callees)} function(s). "
                    f"Part of the active call graph.")

    if snippet:
        with st.expander("Target function snippet"):
            st.code(snippet, language="c")

    # ---------------- Confirm removal gate (conservative + override) ----------------
    force_remove = st.checkbox(
        "Override safety checks (force removal)",
        value=False,
        help="Enable to remove even if static/LLM disagree.",
        key="force_remove_checkbox"
    )
    allow_remove = (static_decision == "SAFE_TO_REMOVE" and llm_decision == "REMOVE") or force_remove

    if allow_remove:
        st.success("You may remove this function.")

        # Debug mode checkbox
        debug_removal = st.checkbox(
            "Show debug info",
            key="debug_removal_checkbox",
            help="Display detailed diagnostic information about the removal process"
        )

        # Show diagnostic info if debug enabled
        if debug_removal:
            with st.expander("Session State Diagnostic", expanded=True):
                st.subheader("Current State")

                cached_src = st.session_state.get("__src_text__")

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Session State:**")
                    if cached_src:
                        st.write(f"✅ Source cached: {len(cached_src):,} chars")
                        st.write(f"✅ Filename: {st.session_state.get('__src_name__', 'N/A')}")
                    else:
                        st.write("❌ No source in session state")

                with col2:
                    st.write("**Target Function:**")
                    st.write(f"Name: `{target}`")
                    if cached_src and target + "(" in cached_src:
                        idx = cached_src.find(target + "(")
                        st.write(f"✅ Found at position: {idx:,}")
                    elif cached_src:
                        st.write(f"❌ NOT found in source")
                    else:
                        st.write("❌ No source to search")

                if cached_src and target + "(" in cached_src:
                    idx = cached_src.find(target + "(")
                    preview_start = max(0, idx - 150)
                    preview_end = min(len(cached_src), idx + 250)
                    st.subheader("Function Context Preview")
                    st.code(cached_src[preview_start:preview_end], language="c")

        if st.button(f"Confirm removal of '{target}'", key="confirm_remove"):
            # CRITICAL: Always use session state source
            base_src_text = st.session_state.get("__src_text__")

            # Safety checks
            if not base_src_text:
                st.error("❌ ERROR: Source code not found in session state!")
                st.write("**Please try:**")
                st.write("1. Re-upload your C source file")
                st.write("2. Click 'Analyze' again")
                st.write("3. Then click 'Confirm removal'")
                st.stop()

            if target not in base_src_text:
                st.error(f"❌ ERROR: Function '{target}' not found in source!")
                st.write(f"Source has {len(base_src_text):,} characters")
                st.write("**Please verify:**")
                st.write("- Function name is spelled correctly")
                st.write("- Function exists in the uploaded source file")
                st.stop()

            # Perform removal
            with st.spinner(f"Removing '{target}'..."):
                new_text, removed, info = _remove_function_from_c_source(
                    base_src_text,
                    target,
                    debug=debug_removal
                )

            if removed:
                # Store results in session state
                st.session_state["removed_func"] = target
                st.session_state["updated_src"] = new_text
                st.session_state["orig_src"] = base_src_text
                st.session_state["src_name"] = st.session_state.get("__src_name__", "source.c")
                st.session_state["removal_info"] = info

                # Save a physical copy to temp directory for download
                try:
                    base, dot, ext = st.session_state["src_name"].rpartition(".")
                    out_name = base + f"_without_{target}" + (dot + ext if dot else ".c")
                    save_path = Path(tempfile.gettempdir()) / out_name
                    Path(save_path).write_text(new_text, encoding="utf-8", errors="ignore")
                    st.session_state["saved_file_path"] = str(save_path)
                except Exception as e:
                    st.session_state["saved_file_path"] = ""
                    st.warning(f"Could not save temp file: {e}")

                st.rerun()  # ensure success panel renders immediately
            else:
                st.error(f"❌ Removal failed: {info}")
                st.write("**Troubleshooting tips:**")
                st.write("- Check if the function name is spelled correctly")
                st.write("- The function might be a macro definition (#define)")
                st.write("- Enable 'Show debug info' above for detailed diagnostics")
                st.write("- The function might have unusual formatting or nested preprocessor directives")
    else:
        st.info(
            "Removal is disabled because either the static graph shows dependencies "
            "or the LLM advised against removal."
        )

# ============================== LLM Usage Panel ==============================
st.divider()
st.subheader("LLM API Usage (this session)")

hist = st.session_state.get("usage_history", [])
if not hist:
    st.caption("No API calls yet.")
else:
    df = pd.DataFrame(hist)

    # Metrics row
    total_calls = len(df)
    total_prompt = int(df["prompt_tokens"].fillna(0).sum())
    total_completion = int(df["completion_tokens"].fillna(0).sum())
    total_all = int(df["total_tokens"].fillna(0).sum() if "total_tokens" in df else (total_prompt + total_completion))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("API calls", total_calls)
    m2.metric("Prompt tokens (Σ)", f"{total_prompt:,}")
    m3.metric("Completion tokens (Σ)", f"{total_completion:,}")
    m4.metric("Total tokens (Σ)", f"{total_all:,}")

    # Display the detailed table (latest first)
    df_disp = df.iloc[::-1].reset_index(drop=True)
    st.dataframe(
        df_disp[["time", "model", "target", "llm_decision", "prompt_tokens", "completion_tokens", "total_tokens"]],
        use_container_width=True,
        hide_index=True
    )

# Diagnostics log (raw agent messages) - only show if we have results
if "last_agent_result" in st.session_state:
    msgs = st.session_state["last_agent_result"].get("messages", [])
    if msgs:
        with st.expander("Agent log"):
            for msg in msgs:
                content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None))
                if content:
                    st.code(content, language="text")
