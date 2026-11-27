"""Shared inference runtime: load model, retrieve tools, generate, parse calls, mock-execute."""
import json, re, os, torch, numpy as np
from tokenizers import Tokenizer
from model import TinyGPT, Config, forward_cached
import gen
from gen import SPECIAL, EOT, tools_block, BY, CAT, APPNAME, object_of

HERE = os.path.dirname(os.path.abspath(__file__))

class Brain:
    def __init__(self, ckpt=os.path.join(HERE, "artifacts/model.pt"), tok=os.path.join(HERE, "artifacts/tokenizer.json"), threads=4):
        torch.set_num_threads(threads)
        self.tok = Tokenizer.from_file(tok)
        sd = torch.load(ckpt, map_location="cpu")
        self.cfg = Config(**sd["cfg"]); self.model = TinyGPT(self.cfg); self.model.load_state_dict(sd["model"]); self.model.eval()
        self.sp = {k: self.tok.token_to_id(v) for k, v in SPECIAL.items()}; self.eot = self.tok.token_to_id(EOT)
        self.id2sp = {v: k for k, v in self.sp.items()}

    def encode_segs(self, segs):
        ids = []
        for role, text in segs:
            ids.append(self.sp[role]); ids += self.tok.encode(text).ids if text else []
            if role in ("call", "assistant"): ids.append(self.eot)
        return ids

    @torch.no_grad()
    def generate_turn(self, ids, max_new=160):
        """Greedy-generate from prompt ids until <|eot|>. Returns list of (role, text) generated segments."""
        ids = ids[-(self.cfg.ctx - max_new):]
        logits, cache = forward_cached(self.model, torch.tensor([ids]), None, 0)
        pos = len(ids); out = []
        for _ in range(max_new):
            nxt = int(torch.argmax(logits[0]))
            out.append(nxt)
            if nxt == self.eot or nxt in (self.sp["user"], self.sp["result"], self.sp["tools"]) or pos >= self.cfg.ctx - 1: break
            logits, cache = forward_cached(self.model, torch.tensor([[nxt]]), cache, pos); pos += 1
        segs, cur_role, cur = [], None, []
        for t in out:
            if t in self.id2sp or t == self.eot:
                if cur_role: segs.append([cur_role, self.tok.decode(cur).strip()])
                cur_role, cur = (self.id2sp.get(t), []) if t != self.eot else (None, [])
                if t == self.eot or self.id2sp.get(t) in ("user", "result", "tools"): cur_role = None; break
            else: cur.append(t)
        if cur_role: segs.append([cur_role, self.tok.decode(cur).strip()])
        return segs

class Retriever:
    """Embeds tool descriptions with potion-base-8M (model2vec, static embeddings - fast on CPU)."""
    def __init__(self, path=None):
        from model2vec import StaticModel
        path = path or next(p for p in [os.path.join(HERE, "retriever/potion-base-8M"), "/workspace/jev-executor/data/potion-base-8M", "minishlab/potion-base-8M"] if not p.startswith("/") or os.path.exists(p))
        try:
            self.m = StaticModel.from_pretrained(path)
            self.docs = [f"{APPNAME[c['app']]} {object_of(c)} {c['name'].replace('_', ' ').lower()}. {c['desc']}" for c in CAT]
            self.E = self._emb(self.docs)
        except Exception as e:
            raise RuntimeError(f"Failed to load the tool retriever from {path!r}: {e}") from e
        if self.E.ndim != 2 or self.E.shape[0] != len(CAT) or not np.isfinite(self.E).all():
            raise RuntimeError(f"Tool retriever embeddings are invalid (shape={getattr(self.E, 'shape', None)}).")
        if float(self.E.std()) < 1e-6:
            raise RuntimeError("Tool retriever embeddings are degenerate (nearly identical vectors). Tool search would be constant.")
        # two very different queries must not produce the same ranking (catches a silent zero/constant fallback)
        a = [c["name"] for c in self.top("weather forecast in lagos", 3)]
        b = [c["name"] for c in self.top("show my latest emails", 3)]
        if a == b:
            raise RuntimeError(f"Tool retriever is not ranking by query (got {a} for both weather and email).")
    def _emb(self, xs):
        e = self.m.encode(xs)
        if not isinstance(e, np.ndarray) or e.size == 0 or not np.isfinite(e).all():
            raise RuntimeError("Tool retriever produced empty or non-finite embeddings.")
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9)
    def top(self, query, k=5, app_boost=True):
        s = self.E @ self._emb([query])[0]
        if app_boost:  # if the user names an app, prefer that app's tools
            ql = query.lower()
            for i, c in enumerate(CAT):
                if APPNAME[c["app"]].lower() in ql or c["app"] in ql.replace(" ", ""): s[i] += 0.15
        return [CAT[i] for i in np.argsort(-s)[:k]]

def parse_call(text):
    try:
        d = json.loads(text)
        if isinstance(d, dict) and "tool" in d and isinstance(d.get("args", {}), dict): return d
    except Exception: pass
    return None

def mock_execute(call):
    """MOCK executor: never touches real apps. Returns realistic fake output."""
    c = BY.get(call["tool"])
    if c is None: return {"ok": False, "error": "unknown_tool"}
    missing = [p["name"] for p in c["params"] if p["required"] and p["name"] not in call.get("args", {})]
    if missing: return {"ok": False, "error": "invalid_argument", "missing": missing}
    return gen.mock_result(c, call["args"])

class Agent:
    """Conversation loop: retrieval -> model -> (call -> mock result)* -> reply."""
    def __init__(self, brain=None, retriever=None, k=5, executor=mock_execute, verbose=True):
        self.brain = brain or Brain(); self.ret = retriever or Retriever(); self.k = k; self.exec = executor
        self.history = []; self.users = []; self.verbose = verbose; self.trace = []
    def _tools(self):
        q = " ".join(self.users[-2:])
        tools = self.ret.top(q, self.k)
        used = [BY[json.loads(t)["tool"]] for r, t in self.history if r == "call" and json.loads(t)["tool"] in BY][-2:]
        names = {c["name"] for c in tools}
        return [c for c in used if c["name"] not in names] + tools
    def send(self, text, max_steps=4):
        self.users.append(text); self.history.append(["user", text]); self.trace.append(["user", text])
        tools = self._tools()
        for _ in range(max_steps + 1):
            segs = [["tools", tools_block(tools)]] + self.history
            ids = self.brain.encode_segs(segs)
            while len(ids) > self.brain.cfg.ctx - 170 and len(self.history) > 2:   # drop oldest turns if too long
                self.history = self.history[2:]; segs = [["tools", tools_block(tools)]] + self.history; ids = self.brain.encode_segs(segs)
            out = self.brain.generate_turn(ids)
            if not out: return "(no output)"
            for role, t in out:
                self.history.append([role, t]); self.trace.append([role, t])
            last = out[-1]
            if last[0] == "call":
                call = parse_call(last[1])
                res = self.exec(call) if call else {"ok": False, "error": "invalid_json"}
                self.history.append(["result", json.dumps(res)]); self.trace.append(["result", json.dumps(res)])
                continue
            if last[0] == "assistant": return last[1]
            return "(model stopped without a reply)"
        return "(stopped after too many tool calls)"
