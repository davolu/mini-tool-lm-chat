"""v2 runtime: same loop as v0 plus
  * constrained decoding of tool calls: tool name must be one of the listed tools, keys must be that tool's params
    (no duplicates), JSON is always valid, and identifier-like string values (emails, ids, names, titles, dates, channels,
    paths, queries) must be verbatim spans of the conversation (user messages + tool results) - a copy/pointer constraint.
    Free text (message bodies) may be generated.
  * a confirmation gate enforced in CODE: a risky tool (send/delete/post/...) never runs unless the user just said yes to a
    confirmation question. Otherwise the call is held and the user is asked.
All execution is MOCKED (runtime.mock_execute)."""
import os, sys, json, re, torch, difflib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, ROOT)
import runtime as R
from runtime import parse_call, mock_execute
from model import forward_cached
import gen as G
sys.path.insert(0, HERE)
import plans as PL      # PL.register_extra() adds the Drive tool used by plans; NOT done here so eval2 cases stay identical to v0

COPY_SEMS = {"email", "id", "time", "channel", "location", "path", "query", "title", "generic"}

class CallGrammar:
    def __init__(self, tools, source):
        self.tools = {c["name"]: c for c in tools}; self.source = source
    def _key_info(self, tool, key):
        p = next(p for p in self.tools[tool]["params"] if p["name"] == key)
        return p["type"], (G.sem(p) in COPY_SEMS)
    def check(self, s):
        """Return (ok, done) for prefix s of the call JSON."""
        i = 0; n = len(s)
        def lit(L):
            nonlocal i
            m = min(len(L), n - i)
            if s[i:i + m] != L[:m]: return False, False
            i += m; return True, m == len(L)
        ok, full = lit('{"tool": "')
        if not ok or not full: return ok, False
        j = s.find('"', i)
        name = s[i:] if j < 0 else s[i:j]
        if j < 0: return any(t.startswith(name) for t in list(self.tools) + ["PLAN"]), False
        if name == "PLAN": return self._plan(s, j)
        if name not in self.tools: return False, False
        i = j
        ok, full = lit('", "args": {')
        if not ok or not full: return ok, False
        used = set(); first = True
        while True:
            if i >= n: return True, False
            if s[i] == "}":
                i += 1
                if i >= n: return True, False
                if s[i] != "}": return False, False
                i += 1
                return (i == n), (i == n)
            if not first:
                ok, full = lit(", ")
                if not ok or not full: return ok, False
                if i >= n: return True, False
            ok, full = lit('"')
            if not full: return ok, False
            j = s.find('"', i)
            allowed = [p["name"] for p in self.tools[name]["params"] if p["name"] not in used]
            key = s[i:] if j < 0 else s[i:j]
            if j < 0: return any(a.startswith(key) for a in allowed), False
            if key not in allowed: return False, False
            used.add(key); i = j + 1; first = False
            ok, full = lit(": ")
            if not full: return ok, False
            typ, copy = self._key_info(name, key)
            ok, i2 = self._value(s, i, typ, copy)
            if not ok: return False, False
            if i2 is None: return True, False
            i = i2
    # ---- strict prefix grammar for PLAN calls. Each helper returns (status, i): "ok" (parsed, new pos), "part" (input ended: valid prefix), "bad"
    def _lit(self, s, i, L):
        m = min(len(L), len(s) - i)
        if s[i:i + m] != L[:m]: return "bad", i
        return ("ok", i + len(L)) if m == len(L) else ("part", i)
    def _name(self, s, i, names):
        st, i = self._lit(s, i, '"')
        if st != "ok": return st, i
        j = s.find('"', i)
        if j < 0: return ("part", i) if any(t.startswith(s[i:]) for t in names) else ("bad", i)
        return ("ok", j + 1) if s[i:j] in names else ("bad", i)
    def _choice(self, s, i, opts):
        rest = s[i:]
        for o in opts:
            if rest.startswith(o): return "ok", i + len(o)
        return ("part", i) if any(o.startswith(rest) for o in opts) else ("bad", i)
    def _pval(self, s, i, typ, copy):
        st = self._value(s, i, typ, copy, ref=True)
        return ("bad", i) if not st[0] else (("part", i) if st[1] is None else ("ok", st[1]))
    def _pargs(self, s, i, tool):
        st, i = self._lit(s, i, "{")
        if st != "ok": return st, i
        used = set(); first = True
        while True:
            if i >= len(s): return "part", i
            if s[i] == "}": return "ok", i + 1
            if not first:
                st, i = self._lit(s, i, ", ")
                if st != "ok": return st, i
            allowed = [p["name"] for p in self.tools[tool]["params"] if p["name"] not in used]
            if not allowed: return "bad", i
            st, i2 = self._name(s, i, allowed)
            if st != "ok": return st, i
            key = s[i + 1:i2 - 1]; used.add(key); i = i2; first = False
            st, i = self._lit(s, i, ": ")
            if st != "ok": return st, i
            typ, copy = self._key_info(tool, key)
            st, i = self._pval(s, i, typ, copy)
            if st != "ok": return st, i
    def _seq(self, s, i, parts):
        for f in parts:
            st, i = f(s, i)
            if st != "ok": return st, i
        return "ok", i
    def _step(self, s, i):
        st, i = self._lit(s, i, '{"tool": ')
        if st != "ok": return st, i
        st, i2 = self._name(s, i, list(self.tools))
        if st != "ok": return st, i
        tool = s[i + 1:i2 - 1]; i = i2
        st, i = self._seq(s, i, [lambda s, i: self._lit(s, i, ', "args": '), lambda s, i: self._pargs(s, i, tool)])
        if st != "ok": return st, i
        st, i = self._choice(s, i, ["}", ', "when": '])
        if st != "ok" or s[i - 1] == "}": return st, i
        L = lambda t: (lambda s, i: self._lit(s, i, t))
        return self._seq(s, i, [L('{"type": "count", "watch": '), lambda s, i: self._name(s, i, list(self.tools)), L(', "op": '),
                                lambda s, i: self._choice(s, i, ['">"', '">="', '"<"']), L(', "value": '), self._int, L("}}")])
    def _int(self, s, i):
        j = i
        while j < len(s) and s[j].isdigit(): j += 1
        if j == len(s): return "part", i
        return ("ok", j) if j > i else ("bad", i)
    def _plan(self, s, j):
        L = lambda t: (lambda s, i: self._lit(s, i, t))
        st, i = self._lit(s, j, '", "args": {"steps": [')
        if st != "ok": return st != "bad", False
        k = 0
        while True:
            st, i = self._step(s, i)
            if st != "ok": return st != "bad", False
            k += 1
            st, i = self._choice(s, i, [", ", "]"] if k < 6 else ["]"])
            if st != "ok": return st != "bad", False
            if s[i - 1] == "]": break
        st, i = self._choice(s, i, ["}}", ', "trigger": '])
        if st != "ok": return st != "bad", False
        if s[i - 1] == "}": return i == len(s), i == len(s)
        st, i = self._lit(s, i, '{"type": ')
        if st != "ok": return st != "bad", False
        st, i = self._choice(s, i, ['"schedule", "every": ', '"event", "watch": '])
        if st != "ok": return st != "bad", False
        if s[i - 2:i] == ": " and "every" in s[i - 10:i]:
            st, i = self._seq(s, i, [lambda s, i: self._pval(s, i, "str", False), L("}}}")])
        else:
            st, i = self._seq(s, i, [lambda s, i: self._name(s, i, list(self.tools)), L(', "match": '), lambda s, i: self._pval(s, i, "str", True), L("}}}")])
        if st != "ok": return st != "bad", False
        return i == len(s), i == len(s)
    def _string(self, s, i, copy, ref=False):
        n = len(s)
        if i >= n: return True, None
        if s[i] != '"': return False, None
        j = i + 1
        while j < n and s[j] != '"':
            if s[j] in "\\\n": return False, None
            j += 1
        content = s[i + 1:j]
        if ref and content.startswith("{"):           # reference to an earlier step's output: {{stepN}}
            if j >= n: return bool(re.fullmatch(r"\{(\{(s(t(e(p(\d(\}\}?)?)?)?)?)?)?)?", content)), None
            return (True, j + 1) if re.fullmatch(r"\{\{step\d\}\}", content) else (False, None)
        if copy and content and not re.search(r"(?<![A-Za-z0-9])" + re.escape(content), self.source): return False, None   # must start at a word boundary
        if j >= n: return True, None
        if copy and not content: return False, None
        if copy and not re.search(r"(?<![A-Za-z0-9])" + re.escape(content) + r"(?![A-Za-z0-9@])", self.source): return False, None   # whole span, not a fragment ("boss" of boss@corp.com)
        return True, j + 1
    def _value(self, s, i, typ, copy, ref=False):
        n = len(s)
        if i >= n: return True, None
        if typ == "bool":
            for L in ("true", "false"):
                m = min(len(L), n - i)
                if s[i:i + m] == L[:m]: return (True, None) if m < len(L) else (True, i + len(L))
            return False, None
        if typ in ("int", "num"):
            j = i
            if j < n and s[j] == "-": j += 1
            while j < n and (s[j].isdigit() or (typ == "num" and s[j] == ".")): j += 1
            if j == n: return True, None
            if j == i or s[j] not in ",}": return False, None
            return True, j
        if typ == "list":
            if s[i] != "[": return False, None
            j = i + 1; firstel = True
            while True:
                if j >= n: return True, None
                if s[j] == "]": return (True, j + 1) if not firstel else (False, None)
                if not firstel:
                    if s[j:j + 2] != ", "[:min(2, n - j)]: return False, None
                    if n - j < 2: return True, None
                    j += 2
                ok, j2 = self._string(s, j, copy, ref)
                if not ok: return False, None
                if j2 is None: return True, None
                j = j2; firstel = False
        return self._string(s, i, copy, ref)

class Brain2(R.Brain):
    def __init__(self, ckpt=os.path.join(HERE, "artifacts/model.pt"), tok=os.path.join(HERE, "artifacts/tokenizer.json"), threads=4):
        super().__init__(ckpt, tok, threads)
        self.vocab = [self.tok.decode([i]) for i in range(self.tok.get_vocab_size())]
        self.special_ids = set(self.sp.values()) | {self.eot, 0}

    @torch.no_grad()
    def generate_turn(self, ids, max_new=400, grammar=None, keep=160, on_call=None):
        ids = ids[-(self.cfg.ctx - keep):]
        logits, cache = forward_cached(self.model, torch.tensor([ids]), None, 0)
        pos = len(ids); out = []; in_call = False; call_txt = ""
        for _ in range(max_new):
            if in_call and grammar is not None:
                ok, done = grammar.check(call_txt)
                if done: nxt = self.eot
                else:
                    nxt = None
                    order = torch.argsort(logits[0], descending=True).tolist()
                    for t in order[:64] + order[64:]:
                        if t in self.special_ids: continue
                        if grammar.check(call_txt + self.vocab[t])[0]: nxt = t; break
                    if nxt is None: nxt = self.eot
                    else: call_txt += self.vocab[nxt]
            else:
                nxt = int(torch.argmax(logits[0]))
            out.append(nxt)
            if nxt == self.sp["call"]:
                in_call = True
                if on_call is not None:
                    think = " ".join(t for r, t in self._segs_from_ids(out) if r == "think")
                    g2 = on_call(think)
                    if g2 is not None: grammar = g2
            if nxt == self.eot or nxt in (self.sp["user"], self.sp["result"], self.sp["tools"]) or pos >= self.cfg.ctx - 1: break
            logits, cache = forward_cached(self.model, torch.tensor([[nxt]]), cache, pos); pos += 1
        return self._segs_from_ids(out)

    def _segs_from_ids(self, out):
        segs, cur_role, cur = [], None, []
        for t in out:
            if t in self.id2sp or t == self.eot:
                if cur_role: segs.append([cur_role, self.tok.decode(cur).strip()])
                cur_role, cur = (self.id2sp.get(t), []) if t != self.eot else (None, [])
                if t == self.eot or self.id2sp.get(t) in ("user", "result", "tools"): cur_role = None; break
            else: cur.append(t)
        if cur_role: segs.append([cur_role, self.tok.decode(cur).strip()])
        return segs

YES_RE = re.compile(r"^\s*(y|yes|yeah|yep|sure|ok|okay|go ahead|do it|confirm|proceed|correct|send it|yes please|👍)\b", re.I)
CONFIRM_Q = re.compile(r"(confirm|go ahead|proceed|shall i|should i|okay to)", re.I)
# Only reuse the previous tool shortlist for real follow-ups (confirmations, an email, a number),
# not for a new request that happens to be under 6 words ("Hi", "Weather in lagos").
FOLLOWUP_RE = re.compile(
    r"^\s*(?:"
    r"(?:y|yes|yeah|yep|sure|ok|okay|no|nope|nah|cancel|never mind|don't|do not|"
    r"go ahead|do it|confirm|proceed|correct|send it|yes please|please do|👍)\b"
    r"|(?:it'?s|it is|use|try)?\s*[\w.+-]+@[\w.-]+\.\w+\s*"
    r"|[\d:./+-]+\s*"
    r")", re.I)
NAMED_TOOL_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")

def _closest_catalog_tools(name, hint="", n=3):
    """Resolve a near-miss tool name (e.g. OPENWEATHER_API_SEARCH_PEOPLE) to real catalog tools of the same app."""
    if name in G.BY: return [name]
    parts = name.split("_")
    if len(parts) < 2: return []
    cands = [t for t in G.BY if t.split("_")[0] == parts[0]]
    if not cands: return []
    hint_toks = set(re.findall(r"[A-Za-z0-9]+", (hint + " " + name).upper()))
    def score(t):
        tp = t.split("_")
        pref = 0
        while pref < len(parts) and pref < len(tp) and parts[pref] == tp[pref]: pref += 1
        write = any(w in t for w in ("DELETE", "UPDATE", "CREATE", "SEND", "POST", "UPLOAD", "REMOVE", "ARCHIVE"))
        read_hint = bool(hint_toks & {"READ", "GET", "LIST", "SEARCH", "FETCH", "SHOW", "CURRENT", "WEATHER", "FIND"})
        return (pref, 0 if (write and read_hint) else 1, len(set(tp) & hint_toks), difflib.SequenceMatcher(None, name, t).ratio())
    cands = [t for t in cands if score(t)[0] >= 2]
    cands.sort(key=score, reverse=True)
    return cands[:n]

def inject_named_catalog_tools(tools, text):
    """If think/intent names a catalog tool (or a near-miss of one) that retrieval missed, add it to the shortlist."""
    names = {c["name"] for c in tools}; added = []
    for n in NAMED_TOOL_RE.findall(text or ""):
        for t in _closest_catalog_tools(n, text):
            if t not in names and t in G.BY:
                tools.append(G.BY[t]); names.add(t); added.append(t)
    return added

def not_connected_reply(call, res):
    tool = (call or {}).get("tool", "")
    c = G.BY.get(tool)
    app = G.APPNAME.get(c["app"], (res or {}).get("app") or "That app") if c else ((res or {}).get("app") or "That app")
    if app and app[0].islower(): app = app[0].upper() + app[1:]
    return f"{app} isn't connected in your Composio account, so I can't run {tool}."

class Agent2(R.Agent):
    """Constrained calls + code-enforced confirmation gate."""
    def __init__(self, brain=None, retriever=None, k=5, executor=mock_execute, constrained=True, gate=True):
        super().__init__(brain or Brain2(), retriever, k, executor)
        self.constrained = constrained; self.gate = gate; self.pending = None
        self.plans = PlanRunner(executor)
    def _confirmed(self):
        users = [t for r, t in self.history if r == "user"]
        asst = [t for r, t in self.history if r == "assistant"]
        return bool(users) and YES_RE.search(users[-1]) and bool(asst) and CONFIRM_Q.search(asst[-1])
    def send(self, text, max_steps=4):
        self.users.append(text); self.history.append(["user", text]); self.trace.append(["user", text])
        if self.pending is not None:      # a risky call is waiting for the user's answer (gate asked in code)
            call, self.pending = self.pending, None
            if YES_RE.search(text):
                self._run(call)
            else:
                msg = "Okay, I won't do it."; self.history.append(["assistant", msg]); self.trace.append(["assistant", msg]); return msg
        tools = self._tools()
        for _ in range(max_steps + 1):
            segs = [["tools", G.tools_block(tools)]] + self.history
            ids = self.brain.encode_segs(segs)
            while len(ids) > self.brain.cfg.ctx - 170 and len(self.history) > 2:
                self.history = self.history[2:]; segs = [["tools", G.tools_block(tools)]] + self.history; ids = self.brain.encode_segs(segs)
            source = "\n".join(t for r, t in self.history if r in ("user", "result"))
            grammar = CallGrammar(tools, source) if self.constrained else None
            def _on_call(think, _tools=tools, _src=source):
                if inject_named_catalog_tools(_tools, think):
                    self._last_tools = _tools
                    return CallGrammar(_tools, _src) if self.constrained else None
                return grammar
            if isinstance(self.brain, Brain2):
                out = self.brain.generate_turn(ids, grammar=grammar, on_call=_on_call)
            else:
                out = self.brain.generate_turn(ids)
            if not out: return "(no output)"
            last = out[-1]
            if last[0] == "call":
                call = parse_call(last[1])
                c = G.BY.get(call["tool"]) if call else None
                for role, t in out[:-1]: self.history.append([role, t]); self.trace.append([role, t])
                if call and call["tool"] == "PLAN":
                    err = validate_plan(call.get("args", {}), tools)
                    risky = [] if err else [st["tool"] for st in call["args"]["steps"] if G.BY[st["tool"]]["risky"]]
                    if self.gate and risky and not self._confirmed():
                        self.pending = call
                        msg = "[gate] " + plan_text(call["args"]) + " Shall I set it up?"
                        self.history.append(["assistant", msg]); self.trace.append(["assistant", msg]); return msg
                    self.history.append(["call", last[1]]); self.trace.append(["call", last[1]])
                    res = {"ok": False, "error": err} if err else self.plans.add(call["args"])
                    self.history.append(["result", json.dumps(res)]); self.trace.append(["result", json.dumps(res)])
                    continue
                if self.gate and c is not None and c["risky"] and not self._confirmed():
                    self.pending = call
                    msg = "[gate] " + G.confirm_text(c, call.get("args", {}))
                    self.history.append(["assistant", msg]); self.trace.append(["assistant", msg]); return msg
                self.history.append(["call", last[1]]); self.trace.append(["call", last[1]])
                res = self.exec(call) if call else {"ok": False, "error": "invalid_json"}
                self.history.append(["result", json.dumps(res)]); self.trace.append(["result", json.dumps(res)])
                if isinstance(res, dict) and res.get("error") == "not_connected":
                    msg = not_connected_reply(call, res)
                    self.history.append(["assistant", msg]); self.trace.append(["assistant", msg])
                    return msg
                continue
            for role, t in out: self.history.append([role, t]); self.trace.append([role, t])
            if last[0] == "assistant": return last[1]
            return "(model stopped without a reply)"
        return "(stopped after too many tool calls)"
    def _run(self, call):
        t = json.dumps(call, ensure_ascii=False)
        self.history.append(["think", f"confirmed -> {call['tool']}"]); self.trace.append(["think", f"confirmed -> {call['tool']}"])
        self.history.append(["call", t]); self.trace.append(["call", t])
        res = self.plans.add(call["args"]) if call["tool"] == "PLAN" else self.exec(call)
        self.history.append(["result", json.dumps(res)]); self.trace.append(["result", json.dumps(res)])


# ---------------- plans: validated, executed and watched by ordinary code (all mocked) ----------------
def validate_plan(a, tools):
    names = {c["name"] for c in tools}
    steps = a.get("steps")
    if not isinstance(steps, list) or not steps: return "plan has no steps"
    for i, st in enumerate(steps):
        if st.get("tool") not in names: return f"step {i+1}: unknown tool"
        ps = {p["name"] for p in G.BY[st["tool"]]["params"]}
        if any(k not in ps for k in st.get("args", {})): return f"step {i+1}: unknown arg"
        w = st.get("when")
        if w is not None and not (w.get("type") == "count" and w.get("watch") in names and w.get("op") in (">", ">=", "<") and isinstance(w.get("value"), int)):
            return f"step {i+1}: bad condition"
    t = a.get("trigger")
    if t is not None and t.get("type") not in ("schedule", "event"): return "bad trigger"
    return None

def plan_text(a):
    out = []
    for i, st in enumerate(a["steps"]):
        c = G.BY[st["tool"]]; ln = f"{i+1}) {G.verb_class(c)} {G.object_of(c)} on {G.APPNAME[c['app']]}"
        if "when" in st: ln += f" - only when {G.object_of(G.BY[st['when']['watch']])} count {st['when']['op']} {st['when']['value']}"
        out.append(ln)
    t = a.get("trigger")
    tr = f" It will run {'every ' + t['every'] if t['type'] == 'schedule' else 'whenever ' + str(t.get('match')) + ' shows up'}." if t else ""
    return "Here's the plan: " + "; ".join(out) + "." + tr

OPF = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b}
class PlanRunner:
    """Runs plan steps in order (substituting {{stepN}} with earlier outputs), and keeps conditional steps / triggers
    as watchers. tick() polls the watched tool (mock) and fires a step once its condition holds."""
    def __init__(self, executor):
        self.exec = executor; self.plans = []; self.log = []
    def _sub(self, args, outs):
        def f(v):
            if isinstance(v, str):
                for k, o in outs.items(): v = v.replace("{{step%d}}" % k, o)
            return v
        return {k: f(v) for k, v in args.items()}
    def _step(self, p, i):
        st = p["a"]["steps"][i]; call = {"tool": st["tool"], "args": self._sub(st.get("args", {}), p["outs"])}
        r = self.exec(call); p["outs"][i + 1] = json.dumps(r.get("data", r))[:200]; p["done"].add(i); self.log.append(["run", call, r]); return r
    def add(self, a):
        p = {"a": a, "outs": {}, "done": set(), "polls": 0}; self.plans.append(p)
        ran = []
        if not a.get("trigger"):
            for i, st in enumerate(a["steps"]):
                if "when" in st: break
                self._step(p, i); ran.append(i + 1)
        waiting = [i + 1 for i, st in enumerate(a["steps"]) if "when" in st]
        return {"ok": True, "data": {"status": "scheduled", "ran_now": ran, "waiting_steps": waiting, "trigger": a["trigger"]["type"] if a.get("trigger") else None}}
    def tick(self, counts=None):
        """One watcher poll. counts: optional {watch_tool: n} to simulate the watched count; otherwise grows by 4 per poll."""
        fired = []
        for p in self.plans:
            p["polls"] += 1
            for i, st in enumerate(p["a"]["steps"]):
                w = st.get("when")
                if w is None or i in p["done"]: continue
                n = (counts or {}).get(w["watch"], 4 * p["polls"])
                self.log.append(["poll", w["watch"], n])
                if OPF[w["op"]](n, w["value"]) and all(j in p["done"] for j in range(i)):
                    self._step(p, i); fired.append((st["tool"], n))
        return fired

# ---------------- tool retrieval for long multi-step requests ----------------
SPLIT_RE = re.compile(r"\b(?:then|after that|and then|afterwards|once|when|whenever|as soon as|only when|every|and)\b|[,.;]", re.I)
def _agent2_tools(self):
    text = self.users[-1]
    if getattr(self, "_last_tools", None) and FOLLOWUP_RE.match(text) and len(text.split()) < 8:
        return self._last_tools                         # short answers ("yes", "it's a@b.com") keep the tool set of the request
    # A new request is retrieved on its own. Joining the previous user message mixed tool lists
    # (weather then "show my emails" used to retrieve OpenWeather + Outlook together).
    q = " ".join(self.users[-2:]) if (len(self.users) >= 2 and FOLLOWUP_RE.match(text)) else text
    parts = [p.strip() for p in SPLIT_RE.split(q) if p and len(p.strip().split()) >= 2]
    if len(parts) >= 2:
        names, tools = set(), []
        for c in self.ret.top(q, 3) + [c for p in parts for c in self.ret.top(p, 2)]:
            if c["name"] not in names: names.add(c["name"]); tools.append(c)
        tools = tools[:9]
    else:
        tools = self.ret.top(q, self.k)
        used = [G.BY[json.loads(t)["tool"]] for r, t in self.history if r == "call" and json.loads(t).get("tool") in G.BY][-2:]
        have = {c["name"] for c in tools}
        tools = [c for c in used if c["name"] not in have] + tools
    self._last_tools = tools
    return tools
Agent2._tools = _agent2_tools
