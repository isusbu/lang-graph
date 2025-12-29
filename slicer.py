# slicer.py
#
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Set, List, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Optional OpenAI SDK; if unavailable, the LLM node will raise a helpful error
try:
    from openai import OpenAI  
except Exception:  
    OpenAI = None  

# -----------------------------------------------------------------------------
# Regex helpers for LLVM `print-callgraph` textual output
# -----------------------------------------------------------------------------
NODE_RX = re.compile(r"^Call graph node for function:\s*'([^']+)'", re.M)
CALL_RX = re.compile(r"^\s*(?:CS<[^>]*>\s+)?calls function '([^']+)'", re.IGNORECASE | re.MULTILINE)
# Parse #uses= to detect if function is referenced elsewhere
USES_RX = re.compile(r"#uses=(\d+)", re.IGNORECASE)

# Patterns to detect function pointer usage and registration
FUNC_PTR_PATTERNS = [
    # SQLite function registration
    re.compile(r'sqlite3_create_function[^(]*\([^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(\w+)', re.IGNORECASE),
    re.compile(r'sqlite3_create_function_v2[^(]*\([^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(\w+)', re.IGNORECASE),
    # Function pointer assignment
    re.compile(r'=\s*&?(\w+)\s*;'),
    # Function tables/arrays
    re.compile(r'\{\s*[^}]*?(\w+)\s*[,}]'),
    # Callback parameters
    re.compile(r'callback[^=]*=\s*&?(\w+)', re.IGNORECASE),
]


@dataclass
class FunctionMetadata:
    """Metadata about a function from call graph analysis."""
    name: str
    uses_count: int = 0  # from #uses=N
    has_callers: bool = False
    callers: Set[str] = None
    is_exported: bool = False
    has_address_taken: bool = False
    
    def __post_init__(self):
        if self.callers is None:
            self.callers = set()


# ------------------------
# Callgraph parsing & BFS
# ------------------------
def extract_all_functions(text: str) -> List[str]:
    names = {m.group(1) for m in NODE_RX.finditer(text)}
    names.discard("<null>")  # ignore pseudo/global if present
    return sorted(names, key=str.lower)


def parse_callgraph_text(text: str) -> Dict[str, Set[str]]:
    """Build a reverse mapping: callee -> set(callers).
    This shows INCOMING edges: who calls each function."""
    callers_map: Dict[str, Set[str]] = {}
    current: Optional[str] = None

    for line in text.splitlines():
        m = NODE_RX.match(line)
        if m:
            current = m.group(1)
            continue
        if current:
            m2 = CALL_RX.match(line)
            if m2:
                callee = m2.group(1)
                callers_map.setdefault(callee, set()).add(current)
    return callers_map


def parse_callees_map(text: str) -> Dict[str, Set[str]]:
    """Build a forward mapping: caller -> set(callees).
    This shows OUTGOING edges: what each function calls."""
    callees_map: Dict[str, Set[str]] = {}
    current: Optional[str] = None

    for line in text.splitlines():
        m = NODE_RX.match(line)
        if m:
            current = m.group(1)
            callees_map.setdefault(current, set())  # Initialize even if no calls
            continue
        if current:
            m2 = CALL_RX.match(line)
            if m2:
                callee = m2.group(1)
                callees_map[current].add(callee)
    return callees_map


def parse_function_metadata(text: str, target: str) -> FunctionMetadata:
    """Extract detailed metadata about a function from call graph."""
    metadata = FunctionMetadata(name=target)
    
    # Find the function's node in the call graph
    pattern = re.compile(
        r"Call graph node for function:\s*'" + re.escape(target) + r"'[^\n]*",
        re.IGNORECASE
    )
    
    match = pattern.search(text)
    if match:
        node_line = match.group(0)
        
        # Extract #uses=N
        uses_match = USES_RX.search(node_line)
        if uses_match:
            metadata.uses_count = int(uses_match.group(1))
    
    return metadata


def check_function_pointer_usage(c_text: str, target: str) -> Dict[str, any]:
    """
    Check if function is used as a function pointer or in callbacks.
    Returns dict with detection results.
    """
    result = {
        "has_address_taken": False,
        "function_registration": [],
        "suspicious_patterns": [],
        "context_snippets": []
    }
    
    # Check for address-of operator
    addr_pattern = re.compile(r'&\s*' + re.escape(target) + r'\b')
    addr_matches = addr_pattern.finditer(c_text)
    
    for match in addr_matches:
        result["has_address_taken"] = True
        start = max(0, match.start() - 100)
        end = min(len(c_text), match.end() + 100)
        snippet = c_text[start:end].replace('\n', ' ')
        result["context_snippets"].append(snippet)
    
    # Check for SQLite function registration
    for pattern in FUNC_PTR_PATTERNS[:2]:  
        matches = pattern.finditer(c_text)
        for match in matches:
            if len(match.groups()) > 0 and match.group(1) == target:
                result["function_registration"].append("sqlite3_create_function")
                start = max(0, match.start() - 50)
                end = min(len(c_text), match.end() + 150)
                snippet = c_text[start:end].replace('\n', ' ')
                result["context_snippets"].append(snippet)
    
    # Check for function name without parentheses (potential pointer usage)
    bare_name_pattern = re.compile(
        r'(?<![a-zA-Z_])' + re.escape(target) + r'(?![a-zA-Z_0-9(])'
    )
    
    # Simple heuristic: find occurrences and check context
    for match in bare_name_pattern.finditer(c_text):
        pos = match.start()
        # Check if it's followed by ( - if so, it's a call, not a pointer
        next_char_pos = match.end()
        if next_char_pos < len(c_text):
            # Skip whitespace
            while next_char_pos < len(c_text) and c_text[next_char_pos].isspace():
                next_char_pos += 1
            
            if next_char_pos < len(c_text) and c_text[next_char_pos] != '(':
                # This might be pointer usage
                result["suspicious_patterns"].append(f"Bare reference at position {pos}")
                # Don't add too many snippets
                if len(result["context_snippets"]) < 5:
                    start = max(0, pos - 80)
                    end = min(len(c_text), pos + 80)
                    snippet = c_text[start:end].replace('\n', ' ')
                    result["context_snippets"].append(snippet)
    
    return result


def get_transitive_callers(callers_map: Dict[str, Set[str]], target: str) -> List[str]:
    """Get all functions that call the target, directly or indirectly (incoming edges)."""
    seen: Set[str] = set()
    work: List[str] = list(callers_map.get(target, set()))
    while work:
        who = work.pop()
        if who in seen:
            continue
        seen.add(who)
        for anc in callers_map.get(who, set()):
            if anc not in seen:
                work.append(anc)
    return sorted(seen)


def get_transitive_callees(callees_map: Dict[str, Set[str]], target: str, max_depth: int = 10) -> List[str]:
    """Get all functions that the target calls, directly or indirectly (outgoing edges).
    Limited by max_depth to avoid infinite loops in case of circular dependencies."""
    seen: Set[str] = set()
    work: List[tuple[str, int]] = [(callee, 1) for callee in callees_map.get(target, set())]
    
    while work:
        who, depth = work.pop()
        if who in seen or depth > max_depth:
            continue
        seen.add(who)
        for callee in callees_map.get(who, set()):
            if callee not in seen:
                work.append((callee, depth + 1))
    return sorted(seen)


# -----------------------------------
# Extract function body from sqlite3.c
# -----------------------------------
FUNC_DEF_RX = re.compile(
    r"(^\w[\w\s\*]+?\b{name}\s*\([^\)]*\)\s*\{)"  # function signature start
    r"([\s\S]*?)"                                         # body
    r"\n\}"                                               # closing brace on a new line (heuristic)
    , re.M
)


def find_function_snippet(c_text: str, name: str, max_chars: int = 12000) -> str:
    pattern = re.compile(FUNC_DEF_RX.pattern.replace('{name}', re.escape(name)), re.M)
    m = pattern.search(c_text)
    if not m:
       
        idx = c_text.find(name + "(")
        if idx == -1:
            return ""  
        start = max(0, idx - 300)
        return c_text[start:start + max_chars]
    snippet = m.group(0)
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "\n/* …snip… */\n"
    return snippet


# --------------------------
# LangGraph state definition
# --------------------------
class State(TypedDict):
    messages: Annotated[List, add_messages]

    # inputs
    cg_path: str           # path to callgraph.txt
    c_path: str            # path to sqlite3.c
    target: str            # target function name
    model: str             # LLM model name (optional; can be empty)

    # internal
    cg_text: str
    c_text: str
    all_funcs: List[str]
    callers_map: Dict[str, Set[str]]  # who calls this function (incoming)
    callees_map: Dict[str, Set[str]]  # who this function calls (outgoing)
    
    # Enhanced metadata
    function_metadata: FunctionMetadata
    pointer_usage: Dict[str, any]
    edge_case_warnings: List[str]

    # outputs
    callers: List[str]                     # direct callers (incoming)
    callers_transitive: List[str]          # transitive callers (incoming)
    callees: List[str]                     # direct callees (outgoing)
    callees_transitive: List[str]          # transitive callees (outgoing)
    decision: str                          # "SAFE_TO_REMOVE" | "DEPENDENT" (static-graph based)
    llm_decision: str                      # "REMOVE" | "DO_NOT_REMOVE"
    llm_rationale: str                     # LLM free-text explanation
    token_usage: Dict[str, Optional[int]]  # OpenAI token usage
    evidence_snippet: str                  # extracted function body snippet


# -------------
# LangGraph nodes
# -------------

def load_files(state: State):
    cg_path = state["cg_path"]
    c_path = state["c_path"]
    with open(cg_path, "r", encoding="utf-8", errors="ignore") as f:
        cg_text = f.read()
    with open(c_path, "r", encoding="utf-8", errors="ignore") as f:
        c_text = f.read()
    funcs = extract_all_functions(cg_text)
    return {
        "cg_text": cg_text,
        "c_text": c_text,
        "all_funcs": funcs,
        "messages": [{
            "role": "system",
            "content": f"Loaded files. Callgraph nodes: {len(funcs)} | C size: {len(c_text):,} bytes"
        }],
    }


def parse_graph(state: State):
    cmap = parse_callgraph_text(state["cg_text"])
    callees = parse_callees_map(state["cg_text"])
    return {
        "callers_map": cmap,
        "callees_map": callees,
        "messages": [{
            "role": "system",
            "content": f"Parsed call graph. Functions with callers: {len(cmap)} | Functions with callees: {len(callees)}"
        }],
    }


def analyze_target(state: State):
    target = (state.get("target") or os.environ.get("TARGET_FUNC", "")).strip()
    
    # INCOMING: Who calls this function
    direct_callers = sorted(state["callers_map"].get(target, set()))
    transitive_callers = get_transitive_callers(state["callers_map"], target)
    has_dependency = bool(direct_callers or transitive_callers)
    
    # OUTGOING: What this function calls
    direct_callees = sorted(state["callees_map"].get(target, set()))
    transitive_callees = get_transitive_callees(state["callees_map"], target)
    
    # Extract enhanced metadata
    metadata = parse_function_metadata(state["cg_text"], target)
    metadata.has_callers = has_dependency
    metadata.callers = set(direct_callers)
    
    # Check for function pointer usage
    pointer_usage = check_function_pointer_usage(state["c_text"], target)
    
    # Detect edge cases and generate warnings
    warnings = []
    
    # Warning 1: #uses > 0 but no callers found
    if metadata.uses_count > 0 and not has_dependency:
        warnings.append(
            f"EDGE CASE: Call graph shows #uses={metadata.uses_count} but static analysis "
            f"found 0 callers. This suggests function pointer usage, callbacks, or exports."
        )
    
    # Warning 2: Function pointer/address taken
    if pointer_usage["has_address_taken"]:
        warnings.append(
            f"EDGE CASE: Function address is taken (& operator found). "
            f"This indicates pointer/callback usage."
        )
    
    # Warning 3: SQLite function registration
    if pointer_usage["function_registration"]:
        warnings.append(
            f"CRITICAL: Function is registered with SQLite via {', '.join(pointer_usage['function_registration'])}. "
            f"Removing this will break SQL queries that call this function!"
        )
    
    # Warning 4: Suspicious patterns
    if pointer_usage["suspicious_patterns"] and len(pointer_usage["suspicious_patterns"]) > 3:
        warnings.append(
            f"EDGE CASE: Found {len(pointer_usage['suspicious_patterns'])} suspicious patterns "
            f"suggesting indirect usage."
        )
    
    # Warning 5: Function with no callers but has dependencies
    if not has_dependency and len(direct_callees) > 0:
        warnings.append(
            f"INFO: Function has 0 callers but calls {len(direct_callees)} function(s) directly. "
            f"This is dead code that can be safely removed (if no edge cases)."
        )
    
    # Adjust decision based on edge cases
    decision = "SAFE_TO_REMOVE" if not has_dependency else "DEPENDENT"
    
    # Override decision if critical edge cases detected
    if pointer_usage["function_registration"] or pointer_usage["has_address_taken"]:
        decision = "DEPENDENT"
        warnings.append("Decision overridden to DEPENDENT due to edge case detection.")
    
    # Extract evidence snippet from the C file for LLM context
    snippet = find_function_snippet(state["c_text"], target)

    msgs = [
        {"role": "system", "content": f"Target: {target}"},
        {"role": "system", "content": f"INCOMING: Direct callers ({len(direct_callers)}): " + (", ".join(direct_callers) if direct_callers else "(none)")},
        {"role": "system", "content": f"OUTGOING: Direct callees ({len(direct_callees)}): " + (", ".join(direct_callees) if direct_callees else "(none)")},
        {"role": "system", "content": f"Call graph metadata: #uses={metadata.uses_count}"},
    ]
    
    # Add edge case warnings to messages
    if warnings:
        msgs.append({"role": "system", "content": "EDGE CASE WARNINGS:\n" + "\n".join(warnings)})
    
    msgs.append({"role": "system", "content": "Static decision: " + ("SAFE_TO_REMOVE" if decision == "SAFE_TO_REMOVE" else "⚠️ DEPENDENT")})
    
    return {
        "target": target,
        "callers": direct_callers,
        "callers_transitive": transitive_callers,
        "callees": direct_callees,
        "callees_transitive": transitive_callees,
        "decision": decision,
        "evidence_snippet": snippet,
        "function_metadata": metadata,
        "pointer_usage": pointer_usage,
        "edge_case_warnings": warnings,
        "messages": msgs,
    }


# -----------------
# LLM decision node
# -----------------

LLM_DECISION_SYSTEM = (
    "You are a senior systems engineer assessing whether a C function can be removed "
    "without breaking build/runtime. You are given:\n"
    "1. Static call-graph analysis (direct callers and callees)\n"
    "2. Function's source code snippet\n"
    "3. Edge case detection results (function pointers, callbacks, registrations)\n"
    "4. Call graph metadata (#uses count)\n\n"
    "CRITICAL EDGE CASES TO CONSIDER:\n"
    "- Functions with #uses > 0 but no callers → likely used via function pointers\n"
    "- sqlite3_create_function() registration → function callable from SQL\n"
    "- Address-of operator (&func) → used as callback/function pointer\n"
    "- Exported symbols → may be called from outside this compilation unit\n"
    "- Macro-generated calls → static analysis misses these\n\n"
    "Return a STRICT JSON object with keys:\n"
    "- decision: 'REMOVE' or 'DO_NOT_REMOVE'\n"
    "- rationale: string explaining your decision\n"
    "- confidence: 'HIGH', 'MEDIUM', or 'LOW'\n"
)

LLM_DECISION_USER_TEMPLATE = (
    "Project: SQLite (amalgamation).\n"
    "Target function: {target}\n\n"
    "=== CALL GRAPH (Direct Relationships Only) ===\n"
    "INCOMING - Direct callers ({n_direct}): {direct}\n"
    "OUTGOING - Direct callees ({n_callees}): {callees}\n\n"
    "=== STATIC ANALYSIS ===\n"
    "Static decision: {static_decision}\n"
    "Call graph metadata: #uses={uses_count}\n\n"
    "=== EDGE CASE DETECTION ===\n"
    "{edge_case_info}\n\n"
    "=== FUNCTION POINTER ANALYSIS ===\n"
    "{pointer_info}\n\n"
    "=== FUNCTION SOURCE ===\n"
    "```c\n{snippet}\n```\n\n"
    "DECISION CRITERIA:\n"
    "- If INCOMING callers = 0 and no edge cases detected → REMOVE (dead code)\n"
    "- If OUTGOING callees > 0 but INCOMING = 0 → REMOVE (dead code, won't affect anything)\n"
    "- If edge cases detected (callbacks, registrations) → DO_NOT_REMOVE\n"
    "- If INCOMING callers > 0 → DO_NOT_REMOVE (actively used)\n\n"
    "Based on the above, should this function be removed?"
)


def llm_decide(state: State):
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. `pip install openai` to enable LLM decisions.")

    client = OpenAI()
    model_name = (state.get("model") or os.environ.get("MODEL", "gpt-4")).strip()

    direct = state.get("callers", [])
    callees = state.get("callees", [])
    static_decision = state.get("decision", "UNKNOWN")
    snippet = state.get("evidence_snippet", "")
    
    # Include edge case information
    metadata = state.get("function_metadata")
    pointer_usage = state.get("pointer_usage", {})
    warnings = state.get("edge_case_warnings", [])
    
    # Format edge case info for LLM
    edge_case_info = "\n".join(warnings) if warnings else "No edge cases detected."
    
    # Format pointer usage info
    pointer_info_parts = []
    if pointer_usage.get("has_address_taken"):
        pointer_info_parts.append("✓ Function address is taken (&func)")
    if pointer_usage.get("function_registration"):
        pointer_info_parts.append(f"✓ Registered with: {', '.join(pointer_usage['function_registration'])}")
    if pointer_usage.get("suspicious_patterns"):
        count = len(pointer_usage["suspicious_patterns"])
        pointer_info_parts.append(f"⚠ {count} suspicious indirect usage patterns found")
    if pointer_usage.get("context_snippets"):
        pointer_info_parts.append(f"\nContext snippets ({len(pointer_usage['context_snippets'])} found):")
        for i, snippet_text in enumerate(pointer_usage["context_snippets"][:3], 1):
            pointer_info_parts.append(f"  {i}. ...{snippet_text[:150]}...")
    
    pointer_info = "\n".join(pointer_info_parts) if pointer_info_parts else "No function pointer usage detected."

    user_msg = LLM_DECISION_USER_TEMPLATE.format(
        target=state.get("target", ""),
        static_decision=static_decision,
        n_direct=len(direct),
        direct=", ".join(direct) if direct else "(none)",
        n_callees=len(callees),
        callees=", ".join(callees) if callees else "(none)",
        uses_count=metadata.uses_count if metadata else 0,
        edge_case_info=edge_case_info,
        pointer_info=pointer_info,
        snippet=snippet if snippet else "(function not found in source)",
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": LLM_DECISION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=500,
    )

    # NEW: capture token usage safely (SDK-dependent)
    usage = getattr(resp, "usage", None)
    token_usage = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
        "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
        "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
    }

    content = resp.choices[0].message.content.strip()

    # Parse LLM response
    decision = "DO_NOT_REMOVE"
    rationale = content
    confidence = "UNKNOWN"
    
    import json as _json
    try:
        data = _json.loads(content)
        decision = str(data.get("decision", decision)).strip().upper()
        rationale = str(data.get("rationale", rationale)).strip()
        confidence = str(data.get("confidence", "MEDIUM")).strip().upper()
        
        if decision not in {"REMOVE", "DO_NOT_REMOVE"}:
            decision = "DO_NOT_REMOVE"
            rationale += " [Invalid decision format, defaulting to DO_NOT_REMOVE]"
    except Exception as e:
        rationale = f"JSON parse error: {e}. Raw: {content}"

    return {
        "llm_decision": decision,
        "llm_rationale": rationale,
        "token_usage": token_usage,
        "messages": [{
            "role": "assistant",
            "content":  (
                f"LLM decision: {decision} (confidence: {confidence})\n"
                f"Reason: {rationale[:400]}\n"
                f"Usage: prompt={token_usage.get('prompt_tokens')} "
                f"completion={token_usage.get('completion_tokens')} "
                f"total={token_usage.get('total_tokens')}"
            )
            
            #f"LLM decision: {decision} (confidence: {confidence})\nReason: {rationale[:400]}"
        }],
    }


# -------------------------------
# Build & export the LangGraph app
# -------------------------------

graph = StateGraph(State)

graph.add_node("load_files", load_files)
graph.add_node("parse_graph", parse_graph)
graph.add_node("analyze_target", analyze_target)
graph.add_node("llm_decide", llm_decide)

# Flow: load_files -> parse_graph -> analyze_target -> llm_decide
graph.add_edge("load_files", "parse_graph")
graph.add_edge("parse_graph", "analyze_target")
graph.add_edge("analyze_target", "llm_decide")

graph.set_entry_point("load_files")
graph.set_finish_point("llm_decide")

# Streamlit / programmatic usage
app = graph.compile()


# ----------------
# Optional: CLI use
# ----------------
if __name__ == "__main__":
    import sys, json
    cg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CG_PATH")
    cp = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("C_PATH")
    tgt = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("TARGET_FUNC")
    mdl = sys.argv[4] if len(sys.argv) > 4 else os.environ.get("MODEL")

    result = app.invoke({
        "cg_path": cg,
        "c_path": cp,
        "target": tgt,
        "model": mdl,
        "messages": [],
    })

    # Enhanced output with edge case info (showing only direct calls)
    output = {
        "target": result.get("target"),
        "static_decision": result.get("decision"),
        "llm_decision": result.get("llm_decision"),
        "incoming": {
            "direct_callers": result.get("callers", []),
            "count": len(result.get("callers", [])),
        },
        "outgoing": {
            "direct_callees": result.get("callees", []),
            "count": len(result.get("callees", [])),
        },
        "rationale": result.get("llm_rationale"),
        "edge_case_warnings": result.get("edge_case_warnings", []),
        "metadata": {
            "uses_count": result.get("function_metadata").uses_count if result.get("function_metadata") else 0,
            "has_address_taken": result.get("pointer_usage", {}).get("has_address_taken", False),
            "is_registered": bool(result.get("pointer_usage", {}).get("function_registration", [])),
        }
    }

    print(json.dumps(output, indent=2))
