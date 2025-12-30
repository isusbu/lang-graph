# cpd_slicer.py
#
# Minimal LangGraph pipeline to test whether an LLM can identify
# cross-parameter dependencies (CPDs) using callgraph + optional docs/code context.
#
from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, Set, List, TypedDict, Optional, Any

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# if unavailable, the LLM node will raise a helpful error
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------------------------------------------------------
# Callgraph parsing 
# -----------------------------------------------------------------------------
NODE_RX = re.compile(r"^Call graph node for function:\s*'([^']+)'", re.M)
CALL_RX = re.compile(
    r"^\s*(?:CS<[^>]*>\s+)?calls function '([^']+)'",
    re.IGNORECASE | re.MULTILINE
)

def parse_callees_map(cg_text: str) -> Dict[str, Set[str]]:
    """Return map: function -> direct callees."""
    lines = cg_text.splitlines()
    out: Dict[str, Set[str]] = {}
    cur: Optional[str] = None
    for line in lines:
        m = NODE_RX.match(line)
        if m:
            cur = m.group(1).strip()
            out.setdefault(cur, set())
            continue
        if cur:
            cm = CALL_RX.match(line)
            if cm:
                callee = cm.group(1).strip()
                if callee:
                    out[cur].add(callee)
    return out

def parse_callers_map(cg_text: str) -> Dict[str, Set[str]]:
    """Return map: function -> direct callers (inverse edges)."""
    callees = parse_callees_map(cg_text)
    callers: Dict[str, Set[str]] = {}
    for caller, cs in callees.items():
        for callee in cs:
            callers.setdefault(callee, set()).add(caller)
    return callers

def _bfs_limited(adj: Dict[str, Set[str]], start: str, max_hops: int = 2, max_nodes: int = 200) -> List[str]:
    """BFS up to max_hops hops; returns nodes in discovery order (excluding start)."""
    if not start:
        return []
    visited = {start}
    frontier = [start]
    res: List[str] = []
    depth = 0
    while frontier and depth < max_hops and len(res) < max_nodes:
        nxt: List[str] = []
        for u in frontier:
            for v in sorted(adj.get(u, set())):
                if v in visited:
                    continue
                visited.add(v)
                res.append(v)
                nxt.append(v)
                if len(res) >= max_nodes:
                    break
            if len(res) >= max_nodes:
                break
        frontier = nxt
        depth += 1
    return res

# -----------------------------------------------------------------------------
# Minimal code/doc snippet extraction
# -----------------------------------------------------------------------------
def extract_keyword_windows(text: str, keywords: List[str], window: int = 2, max_lines: int = 120) -> str:
    """Extract lines around occurrences of any keyword. Best-effort; safe on empty text."""
    if not text or not keywords:
        return ""
    lines = text.splitlines()
    key_rx = re.compile("|".join(re.escape(k) for k in keywords if k), re.IGNORECASE)
    hits = [i for i, ln in enumerate(lines) if key_rx.search(ln)]
    if not hits:
        return ""
    ranges = []
    for i in hits:
        lo = max(0, i - window)
        hi = min(len(lines), i + window + 1)
        ranges.append((lo, hi))
    # merge overlaps
    ranges.sort()
    merged = []
    for lo, hi in ranges:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    # emit
    out_lines: List[str] = []
    for lo, hi in merged:
        out_lines.extend(lines[lo:hi])
        out_lines.append("")  # separator
        if len(out_lines) >= max_lines:
            break
    return "\n".join(out_lines[:max_lines]).strip()

def find_function_snippet(src: str, func: str, max_chars: int = 5000) -> str:
    """
    Best-effort C function body extractor.
    Returns up to max_chars characters.
    """
    if not src or not func:
        return ""
    # Try to find "<ret> func(" style
    pat = re.compile(r"^[\t ]*[A-Za-z_][\w\s\*\(\),]*\b" + re.escape(func) + r"\s*\(", re.M)
    m = pat.search(src)
    if not m:
        return ""
    i = m.start()
    # Find opening brace
    brace = src.find("{", m.end())
    if brace == -1:
        return src[i: m.end()][:max_chars]
    depth = 0
    j = brace
    while j < len(src):
        ch = src[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                j += 1
                break
        j += 1
    snippet = src[i:j]
    return snippet[:max_chars]

# -----------------------------------------------------------------------------
# CPD testcases selected from  FAST'23 ConfD paper
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class CPDTestcase:
    id: str
    component: str         # e.g., "mke2fs", "mkfs.xfs", "ext4 mount"
    param_a: str
    param_b: str
    relation: str          # requires | conflicts | value_constraint
    direction: str         # A->B | B->A | mutual
    note: str = ""

# Ground truth from paper examples (used for evaluation, NOT shown to LLM)
TESTCASES: Dict[str, CPDTestcase] = {
    "CPD-1": CPDTestcase(
        id="CPD-1",
        component="mke2fs",
        param_a="bigalloc",
        param_b="blocksize",
        relation="requires",
        direction="A->B",
        note="bigalloc depends on blocksize setting"
    ),
    "CPD-2": CPDTestcase(
        id="CPD-2",
        component="mke2fs",
        param_a="resize_inode",
        param_b="sparse_super",
        relation="requires",
        direction="A->B",
        note="resize_inode requires sparse_super"
    ),
    "CPD-3": CPDTestcase(
        id="CPD-3",
        component="mke2fs",
        param_a="meta_bg",
        param_b="resize_inode",
        relation="conflicts",
        direction="mutual",
        note="meta_bg cannot be used together with resize_inode"
    ),
    
}

def list_testcases() -> List[Dict[str, str]]:
    """Convenience for UI."""
    out = []
    for k in sorted(TESTCASES.keys()):
        tc = TESTCASES[k]
        out.append({
            "id": tc.id,
            "component": tc.component,
            "param_a": tc.param_a,
            "param_b": tc.param_b,
            "relation": tc.relation,
            "direction": tc.direction,
            "note": tc.note,
        })
    return out

# -----------------------------------------------------------------------------
# LangGraph State
# -----------------------------------------------------------------------------
class State(TypedDict, total=False):
    # Inputs
    cg_path: str
    doc_path: str
    code_path: str
    testcase_id: str
    anchor_functions: str     # comma-separated
    context_mode: str         # docs_only | cg_only | docs_cg | docs_cg_code
    max_hops: int
    model: str

    # Loaded contents
    cg_text: str
    doc_text: str
    c_text: str

    # Parsed graph
    callers_map: Dict[str, Set[str]]
    callees_map: Dict[str, Set[str]]

    # Ground truth
    component: str
    param_a: str
    param_b: str
    gt_relation: str
    gt_direction: str
    gt_note: str

    # Context for LLM
    doc_context: str
    cg_context: str
    code_context: str

    # LLM outputs
    pred_relation: str
    pred_direction: str
    confidence: Optional[float]
    rationale: str
    evidence: List[str]
    llm_raw: str
    token_usage: Dict[str, Optional[int]]

    # Evaluation
    match: bool

    # Logs/messages
    messages: Any

# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------
def load_inputs(state: State) -> State:
    def _read(path: str) -> str:
        if not path:
            return ""
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    cg_text = _read(state.get("cg_path", ""))
    doc_text = _read(state.get("doc_path", ""))
    c_text = _read(state.get("code_path", ""))

    tc_id = (state.get("testcase_id") or "").strip()
    tc = TESTCASES.get(tc_id)
    if not tc:
        raise ValueError(f"Unknown testcase_id='{tc_id}'. Expected one of: {', '.join(sorted(TESTCASES.keys()))}")

    msgs = [
        {"role": "system", "content": f"Loaded inputs. callgraph_chars={len(cg_text)} docs_chars={len(doc_text)} code_chars={len(c_text)}"},
        {"role": "system", "content": f"Testcase={tc.id} component={tc.component} A={tc.param_a} B={tc.param_b}"},
    ]
    return {
        "cg_text": cg_text,
        "doc_text": doc_text,
        "c_text": c_text,
        "component": tc.component,
        "param_a": tc.param_a,
        "param_b": tc.param_b,
        "gt_relation": tc.relation,
        "gt_direction": tc.direction,
        "gt_note": tc.note,
        "messages": msgs,
    }

def parse_graph(state: State) -> State:
    cg_text = state.get("cg_text", "")
    callers_map = parse_callers_map(cg_text) if cg_text else {}
    callees_map = parse_callees_map(cg_text) if cg_text else {}
    return {
        "callers_map": callers_map,
        "callees_map": callees_map,
        "messages": [
            {"role": "system", "content": f"Parsed callgraph: nodes_with_callers={len(callers_map)} nodes_with_callees={len(callees_map)}"}
        ],
    }

def build_context(state: State) -> State:
    mode = (state.get("context_mode") or "docs_cg").strip()
    max_hops = int(state.get("max_hops") or 2)

    param_a = state.get("param_a", "")
    param_b = state.get("param_b", "")
    component = state.get("component", "")

    doc_text = state.get("doc_text", "")
    c_text = state.get("c_text", "")

    # docs context
    doc_ctx = ""
    if mode in ("docs_only", "docs_cg", "docs_cg_code"):
        # common aliases so docs-only works with man-page wording
        keywords = [param_a, param_b]
        
         # CPD-1 special case: man page uses "block size" / "-b" not always "blocksize"
        if param_b.lower() == "blocksize":
            keywords += ["block size", "block-size", "-b", "cluster-size", "-C"]
        doc_ctx = extract_keyword_windows(doc_text, keywords, window=5, max_lines=180)
       # doc_ctx = extract_keyword_windows(doc_text, [param_a, param_b], window=3, max_lines=140)

    # callgraph context
    cg_ctx = ""
    if mode in ("cg_only", "docs_cg", "docs_cg_code"):
        anchors_raw = (state.get("anchor_functions") or "").strip()
        anchors = [a.strip() for a in anchors_raw.split(",") if a.strip()]
        callers_map = state.get("callers_map", {})
        callees_map = state.get("callees_map", {})

        chunks: List[str] = []
        if not anchors:
            chunks.append("No anchor function provided. Callgraph context is empty.")
        else:
            for fn in anchors[:8]:
                direct_callers = sorted(callers_map.get(fn, set()))
                direct_callees = sorted(callees_map.get(fn, set()))
                trans_callers = _bfs_limited(callers_map, fn, max_hops=max_hops, max_nodes=120)
                trans_callees = _bfs_limited(callees_map, fn, max_hops=max_hops, max_nodes=120)
                chunks.append(
                    "\n".join([
                        f"[ANCHOR] {fn}",
                        f"  Direct callers ({len(direct_callers)}): " + (", ".join(direct_callers[:50]) if direct_callers else "(none)"),
                        f"  Direct callees ({len(direct_callees)}): " + (", ".join(direct_callees[:50]) if direct_callees else "(none)"),
                        f"  Transitive callers<=hops{max_hops} ({len(trans_callers)}): " + (", ".join(trans_callers[:80]) if trans_callers else "(none)"),
                        f"  Transitive callees<=hops{max_hops} ({len(trans_callees)}): " + (", ".join(trans_callees[:80]) if trans_callees else "(none)"),
                    ])
                )
        cg_ctx = "\n\n".join(chunks).strip()

    # code context
    code_ctx = ""
    if mode == "docs_cg_code":
        # Prefer keyword windows; if anchors exist, also include function snippets.
        kw_ctx = extract_keyword_windows(c_text, [param_a, param_b], window=3, max_lines=160)
        anchors_raw = (state.get("anchor_functions") or "").strip()
        anchors = [a.strip() for a in anchors_raw.split(",") if a.strip()]
        snippets: List[str] = []
        for fn in anchors[:4]:
            sn = find_function_snippet(c_text, fn, max_chars=2500)
            if sn:
                snippets.append(f"// --- snippet: {fn} ---\n{sn}")
        parts = []
        if kw_ctx:
            parts.append("// --- keyword windows ---\n" + kw_ctx)
        if snippets:
            parts.append("\n\n".join(snippets))
        code_ctx = "\n\n".join(parts).strip()

    msgs = [
        {"role": "system", "content": f"Built context for component={component}. mode={mode} max_hops={max_hops}"},
        {"role": "system", "content": f"doc_ctx_chars={len(doc_ctx)} cg_ctx_chars={len(cg_ctx)} code_ctx_chars={len(code_ctx)}"},
    ]
    return {"doc_context": doc_ctx, "cg_context": cg_ctx, "code_context": code_ctx, "messages": msgs}

def _normalize_relation(rel: str) -> str:
    r = (rel or "").strip().lower()
    if r in ("requires", "require", "dependency", "depends", "depends_on", "dependent"):
        return "requires"
    if r in ("conflict", "conflicts", "incompatible", "mutex", "mutually_exclusive"):
        return "conflicts"
    if r in ("value_constraint", "constraint", "range", "bounded", "value-dependent"):
        return "value_constraint"
    if r in ("unknown", "none", "n/a", ""):
        return "unknown"
    # best-effort mapping
    if "conf" in r or "mutex" in r:
        return "conflicts"
    if "require" in r or "depend" in r:
        return "requires"
    if "constraint" in r or "range" in r or "value" in r:
        return "value_constraint"
    return "unknown"

def _normalize_direction(dirn: str) -> str:
    d = (dirn or "").strip().upper().replace(" ", "")
    if d in ("A->B", "A>B", "A2B", "A→B"):
        return "A->B"
    if d in ("B->A", "B>A", "B2A", "B→A"):
        return "B->A"
    if d in ("MUTUAL", "BOTH", "A<->B", "A↔B", "BIDIRECTIONAL"):
        return "mutual"
    # parse common phrases
    if "A" in d and "B" in d and ("<->" in d or "BOTH" in d or "MUTUAL" in d):
        return "mutual"
    if "A" in d and "B" in d and ("A" in d.split("->")[0] if "->" in d else False):
        return "A->B"
    return "unknown"

LLM_SYSTEM = (
    "You are a configuration dependency analyst.\n"
    "Given limited evidence (docs excerpts, callgraph neighborhoods, code snippets), infer whether there is a\n"
    "CROSS-PARAMETER DEPENDENCY (CPD) between parameter A and parameter B.\n\n"
    "Return STRICT JSON (no markdown) with keys:\n"
    "  relation: one of [\"requires\", \"conflicts\", \"value_constraint\", \"unknown\"]\n"
    "  direction: one of [\"A->B\", \"B->A\", \"mutual\", \"unknown\"]\n"
    "  confidence: number from 0.0 to 1.0\n"
    "  rationale: short explanation\n"
    "  evidence: array of short bullet strings (quote or reference snippets provided; do NOT invent sources)\n"
)

def llm_predict(state: State) -> State:
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Install `openai` to enable LLM predictions.")

    client = OpenAI()
    model_name = (state.get("model") or os.environ.get("MODEL", "gpt-4o-mini")).strip()

    component = state.get("component", "")
    param_a = state.get("param_a", "")
    param_b = state.get("param_b", "")
    mode = (state.get("context_mode") or "docs_cg").strip()

    doc_ctx = state.get("doc_context", "")
    cg_ctx = state.get("cg_context", "")
    code_ctx = state.get("code_context", "")

    user_parts = [
        f"Component: {component}",
        f"Parameter A: {param_a}",
        f"Parameter B: {param_b}",
        f"Context mode: {mode}",
        "",
        "=== DOCS CONTEXT (may be empty) ===",
        doc_ctx if doc_ctx else "(empty)",
        "",
        "=== CALLGRAPH CONTEXT (may be empty) ===",
        cg_ctx if cg_ctx else "(empty)",
        "",
        "=== CODE CONTEXT (may be empty) ===",
        code_ctx if code_ctx else "(empty)",
        "",
        "Task: Infer CPD between A and B and output the JSON schema exactly."
    ]
    user_msg = "\n".join(user_parts)

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )

    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
    }

    # Parse JSON robustly
    pred_relation = "unknown"
    pred_direction = "unknown"
    confidence = None
    rationale = content
    evidence: List[str] = []

    parsed: Optional[dict] = None
    try:
        parsed = json.loads(content)
    except Exception:
        # Try to salvage JSON object from text
        m = re.search(r"\{.*\}", content, re.S)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        pred_relation = _normalize_relation(str(parsed.get("relation", "")))
        pred_direction = _normalize_direction(str(parsed.get("direction", "")))
        try:
            confidence = float(parsed.get("confidence", None))
        except Exception:
            confidence = None
        rationale = str(parsed.get("rationale", rationale)).strip()
        ev = parsed.get("evidence", [])
        if isinstance(ev, list):
            evidence = [str(x).strip() for x in ev if str(x).strip()][:12]

    msgs = [
        {"role": "assistant", "content": content},
        {"role": "system", "content": f"TOKENS: prompt={token_usage.get('prompt_tokens')} completion={token_usage.get('completion_tokens')} total={token_usage.get('total_tokens')}"},
    ]
    return {
        "pred_relation": pred_relation,
        "pred_direction": pred_direction,
        "confidence": confidence,
        "rationale": rationale,
        "evidence": evidence,
        "llm_raw": content,
        "token_usage": token_usage,
        "messages": msgs,
    }

def evaluate(state: State) -> State:
    gt_rel = _normalize_relation(state.get("gt_relation", "unknown"))
    gt_dir = _normalize_direction(state.get("gt_direction", "unknown"))

    pr = _normalize_relation(state.get("pred_relation", "unknown"))
    pd = _normalize_direction(state.get("pred_direction", "unknown"))

    match = (pr == gt_rel) and (pd == gt_dir)

    msgs = [
        {"role": "system", "content": f"Evaluation: predicted=({pr},{pd}) ground_truth=({gt_rel},{gt_dir}) match={match}"},
    ]
    return {"match": match, "messages": msgs}

# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
graph = StateGraph(State)
graph.add_node("load_inputs", load_inputs)
graph.add_node("parse_graph", parse_graph)
graph.add_node("build_context", build_context)
graph.add_node("llm_predict", llm_predict)
graph.add_node("evaluate", evaluate)

graph.set_entry_point("load_inputs")
graph.add_edge("load_inputs", "parse_graph")
graph.add_edge("parse_graph", "build_context")
graph.add_edge("build_context", "llm_predict")
graph.add_edge("llm_predict", "evaluate")
graph.set_finish_point("evaluate")

app = graph.compile()
