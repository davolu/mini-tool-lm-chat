"""Executors for the v2 agent: MockExecutor (fake results) and LiveExecutor (real Composio calls).

Safety design (for the Streamlit app):
  * The Composio key is read ONLY from the process environment (COMPOSIO_API_KEY); it is never printed, logged or written.
    No auth configs or connected accounts are created - only existing ACTIVE accounts are used.
  * Read-only tools (first verb after the app prefix is list/get/search/fetch/find/...) run directly.
    Everything else (send, post, create, update, upload, delete, forward, move, unknown verbs...) is HELD in `pending`
    and runs only when approve(id) is called (the UI's Approve button), with the exact args shown to the user.
  * Tools whose app has no ACTIVE Composio account return a clear {"ok": false, "error": "not_connected", ...}.
  * connect_link(toolkit) may return a Composio Connect URL so the user can authorize that app. It reuses an existing
    auth config, or creates a Composio-managed auth config only when none exists and managed auth is available. It never
    creates custom/paid auth configs, never completes OAuth, and never waits for the user to finish.
  * Nothing is written to disk (no logs of args/results)."""
import os, re, json, time, itertools, threading
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import sys; sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import gen as G
from runtime import mock_execute

READ_WORDS = {"GET", "FETCH", "LIST", "SEARCH", "READ", "FIND", "LOAD", "RETRIEVE", "LOOKUP", "COUNT", "CHECK", "QUERY", "DESCRIBE", "VIEW", "WHO", "WEATHER"}
WRITE_WORDS = {"SEND", "POST", "CREATE", "DELETE", "TRASH", "UPDATE", "REPLY", "PUBLISH", "UPLOAD", "ADD", "MOVE", "REMOVE", "INSERT", "APPEND", "PATCH",
               "MODIFY", "SHARE", "FORWARD", "SCHEDULE", "ARCHIVE", "BATCH", "IMPORT", "WRITE", "EDIT", "COPY", "INVITE", "MARK", "SET", "CANCEL", "CLEAR",
               "RENAME", "PUT", "BLOCK", "LIKE", "FOLLOW", "UNFOLLOW", "PIN", "SUBMIT", "RUN", "TRIGGER", "EXECUTE", "GENERATE", "EMPTY", "QUICK", "DOWNLOAD",
               "REVOKE", "ENABLE", "DISABLE", "SIGN", "PAY", "CHARGE", "REFUND", "ORDER", "BOOK", "SYNC", "ACCEPT", "DECLINE", "RESPOND", "SUBSCRIBE", "UNSUBSCRIBE", "START", "STOP"}
NO_AUTH_APPS = {"composio_search", "hackernews"}          # verified against Composio: these toolkits need no account
PLATFORM_CONNECT = "https://platform.composio.dev"

def scrub(text):
    """Remove the API key (if it ever appears in an error message) before any text leaves this module."""
    t = str(text); k = os.environ.get("COMPOSIO_API_KEY")
    return t.replace(k, "***") if k else t

def toolkit_of(slug):
    """Catalog slug -> Composio toolkit slug (the catalog came from real Composio slugs, so the app is the slug prefix)."""
    c = G.BY.get(slug)
    return c["app"] if c else slug.split("_")[0].lower()

def classify(slug):
    """'read' only when the first verb after the app prefix is a read verb AND no write verb appears anywhere in the slug.
    Everything else, including unknown verbs and catalog tools flagged risky, is 'risky' (err toward risky)."""
    c = G.BY.get(slug)
    if c is not None and c.get("risky"): return "risky"
    tk = toolkit_of(slug)
    app = tk.upper() + "_"
    s = slug.upper(); s = s[len(app):] if (s.startswith(app) and tk != "composio_search") else s   # COMPOSIO_SEARCH_*: "SEARCH" is the verb
    words = s.split("_")
    if any(w in WRITE_WORDS for w in words): return "risky"
    if words and (words[0] in READ_WORDS or any(w in READ_WORDS for w in words)): return "read"
    return "risky"

def _short(v, n=40):
    s = str(v).replace("\n", " ")
    return s if len(s) <= n else s[:n - 1] + "…"

PREF = [("id", "id"), ("messageId", "id"), ("objectID", "id"), ("subject", "subject"), ("title", "title"), ("name", "name"), ("summary", "title"),
        ("from", "from"), ("sender", "from"), ("snippet", "snippet"), ("date", "date")]
def _lists(d, path=""):
    if isinstance(d, list): yield path, d
    elif isinstance(d, dict):
        for k, v in d.items(): yield from _lists(v, path + "/" + str(k))
def _item(it):
    """Map a real item to the few short keys the model saw in training (id / from / subject / name / title / snippet)."""
    out = {}
    for src, dst in PREF:
        if src in it and dst not in out and isinstance(it[src], (str, int, float)) and str(it[src]).strip():
            out[dst] = _short(it[src], 10 if dst == "id" else 36)
    if "id" in out and len(out) > 2: out.pop("id")                       # ids are rarely needed in replies; keep room for text fields
    if len(out) < 1: out = {k: _short(v, 30) for k, v in it.items() if isinstance(v, (str, int, float))}
    return dict(list(out.items())[:3])
def compact(resp, max_chars=480):
    """Shrink a real Composio response to the mock-result shape the model was trained on:
    {"ok": true, "data": {"items": [{...up to 3 items, a few short fields...}]}}; context is only 640 tokens."""
    if not resp.get("ok"):
        out = {"ok": False, "error": _short(resp.get("error") or "failed", 120)}
        for k in ("app", "missing"):
            if k in resp: out[k] = resp[k]
        return out
    data = resp.get("data")
    cand = sorted(_lists(data), key=lambda pv: -len(pv[1])) if data is not None else []
    cand = [(p, l) for p, l in cand if l and isinstance(l[0], dict)]
    if cand:
        path, lst = cand[0]
        out = {"ok": True, "data": {"items": [_item(it) for it in lst[:3]]}}
        if len(lst) > 3: out["data"]["count"] = len(lst)
    else:
        flat = {k: _short(v) for k, v in (data.items() if isinstance(data, dict) else [("result", data)]) if not isinstance(v, (dict, list))}
        out = {"ok": True, "data": dict(list(flat.items())[:4])}
    s = json.dumps(out, ensure_ascii=False)
    while len(s) > max_chars and out["data"].get("items"):
        out["data"]["items"].pop(); s = json.dumps(out, ensure_ascii=False)
    return out

def count_items(resp):
    """Best-effort item count of a real response: length of the longest list of dicts in it (None if there is none)."""
    if not resp.get("ok"): return None
    c = [l for _, l in _lists(resp.get("data")) if l and isinstance(l[0], dict)]
    return max((len(l) for l in c), default=0) if c else 0

class MockExecutor:
    """Fake results only (v2 default). Same interface as LiveExecutor."""
    mode = "mock"
    def __init__(self): self.events = []; self.pending = []
    def classify(self, slug): return classify(slug)
    def __call__(self, call):
        r = mock_execute(call); self.events.append({"call": call, "full": r, "kind": classify(call.get("tool", "")), "status": "mock", "via": getattr(self, "_via", "call")}); return r
    def count(self, slug, polls=1): return 4 * polls, "simulated count (mock mode)"
    def connected(self): return {}
    def run_step(self, call, source="plan", meta=None):
        self._via = "plan"; r = self(call); self._via = "call"; return "ran", r, r

class LiveExecutor:
    mode = "live"
    def __init__(self, user_ids=None, composio=None):
        self._c = composio; self._lock = threading.Lock(); self._ids = itertools.count(1)
        self.user_ids = [u.strip() for u in (user_ids or os.environ.get("COMPOSIO_USER_IDS") or os.environ.get("COMPOSIO_USER_ID") or "").split(",") if u.strip()] \
            if not isinstance(user_ids, (list, tuple)) else list(user_ids)
        self.pending, self.events = [], []
        self._owner, self._conn_cache, self._ver = {}, None, {}
    # ---- availability / accounts (read-only) -------------------------------------------------
    @staticmethod
    def has_key(): return bool(os.environ.get("COMPOSIO_API_KEY"))
    @property
    def composio(self):
        if self._c is None:
            from composio import Composio
            self._c = Composio()                        # reads COMPOSIO_API_KEY from the environment; the key never passes through our code
        return self._c
    def _accounts(self):
        """{toolkit: [user_id, ...]} of ACTIVE connected accounts (existing ones only)."""
        if self._conn_cache is None:
            m = {}
            r = self.composio.connected_accounts.list(statuses=["ACTIVE"], limit=100)
            for i in r.items: m.setdefault(i.toolkit.slug, []).append(getattr(i, "user_id", None))
            self._conn_cache = m
        return self._conn_cache
    def refresh(self): self._conn_cache = None; self._owner.clear()
    def connect_link(self, toolkit):
        """Return (url_or_None, reason) so the user can connect `toolkit` under the auto-picked user id.

        Never completes OAuth. Never creates a custom/paid auth config. A Composio-managed auth config is created
        only when the toolkit has no existing config and Composio-managed auth is available. The result is cached
        per toolkit so repeat questions do not start extra connection requests.
        """
        if not toolkit or toolkit in NO_AUTH_APPS: return None, "no_auth_needed"
        cache = getattr(self, "_connect_cache", None)
        if cache is None: cache = self._connect_cache = {}
        if toolkit in cache: return cache[toolkit]
        try:
            url, reason = self._make_connect_link(toolkit)
        except Exception as e:
            url, reason = None, scrub(str(e))[:200]
        cache[toolkit] = (url, reason)
        return url, reason
    def _make_connect_link(self, toolkit):
        uid = (self.user_ids or self.default_user_ids() or [None])[0]
        if not uid: return None, "no_user_id"
        cfgs = self.composio.auth_configs.list(toolkit_slug=toolkit, limit=100)
        items = list(getattr(cfgs, "items", None) or [])
        def dump(it): return it.model_dump() if hasattr(it, "model_dump") else {}
        # Prefer an existing Composio-managed config; newest first. Never create a custom config.
        items.sort(key=lambda it: (0 if dump(it).get("is_composio_managed") else 1, str(dump(it).get("created_at") or "")), reverse=False)
        managed = [it for it in items if dump(it).get("is_composio_managed")]
        existing = managed or items
        auth_id = None
        if existing:
            # newest: items were sorted oldest-first by created_at within each managed flag, so take last of the preferred group
            existing.sort(key=lambda it: str(dump(it).get("created_at") or ""))
            auth_id = existing[-1].id
        else:
            info = self.composio.toolkits.get(toolkit)
            d = info.model_dump() if hasattr(info, "model_dump") else {}
            schemes = d.get("composio_managed_auth_schemes") or d.get("composio_managed_auth") or []
            if not schemes:
                return None, "no_managed_auth"
            # Composio-managed auth is available and no config exists yet: create only that managed config.
            created = self.composio._client.auth_configs.create(
                toolkit={"slug": toolkit},
                auth_config={"type": "use_composio_managed_auth", "tool_access_config": {"tools_for_connected_account_creation": []}},
            )
            auth_id = getattr(getattr(created, "auth_config", created), "id", None) or getattr(created, "id", None)
            if not auth_id: return None, "managed_create_failed"
        req = self.composio.connected_accounts.link(user_id=uid, auth_config_id=auth_id, allow_multiple=True)
        url = getattr(req, "redirect_url", None)
        if not url: return None, "no_redirect_url"
        return url, "ok"
    def default_user_ids(self):
        """If no user id is configured: the user id that owns the most ACTIVE accounts (the main account), then the rest by count."""
        cnt = {}
        for tk, uids in self._accounts().items():
            for u in uids:
                if u: cnt[u] = cnt.get(u, 0) + 1
        return [u for u, _ in sorted(cnt.items(), key=lambda kv: -kv[1])]
    def owner(self, toolkit):
        if toolkit in self._owner: return self._owner[toolkit]
        acc = self._accounts().get(toolkit, [])
        order = self.user_ids or self.default_user_ids()
        o = next((u for u in order if u in acc), None)
        self._owner[toolkit] = o; return o
    def connected(self):
        """Names of apps (toolkits) that have an active account under the configured user ids, plus no-auth apps."""
        order = self.user_ids or self.default_user_ids()
        names = sorted(tk for tk, uids in self._accounts().items() if any(u in uids for u in order))
        return {"connected": names, "no_auth": sorted(NO_AUTH_APPS)}
    def classify(self, slug): return classify(slug)
    # ---- running tools ----------------------------------------------------------------------------
    def _check(self, call):
        slug = call.get("tool", ""); c = G.BY.get(slug)
        if c is None: return {"ok": False, "error": "unknown_tool"}
        missing = [p["name"] for p in c["params"] if p["required"] and p["name"] not in call.get("args", {})]
        if missing: return {"ok": False, "error": "invalid_argument", "missing": missing}
        return None
    def run_real(self, call, user_override=None, small_pages=True):
        """Execute one call against Composio NOW (used for read-only calls and for approved calls). Returns a full result dict."""
        bad = self._check(call)
        if bad: return bad
        slug = call["tool"]; tk = toolkit_of(slug)
        call = {"tool": slug, "args": dict(call.get("args", {}))}
        if small_pages and classify(slug) == "read":      # the model rarely sets a page size: keep read results small
            for p in G.BY[slug]["params"]:
                if p["name"] in ("max_results", "maxResults", "limit", "page_size", "pageSize", "size") and p["name"] not in call["args"]:
                    call["args"][p["name"]] = 5; break
        uid = user_override or self.owner(tk)
        if uid is None and tk not in NO_AUTH_APPS:
            return {"ok": False, "error": "not_connected", "app": tk, "message": f"No active Composio account for '{tk}' - connect it in Composio first (this app never creates connections)."}
        uid = uid or (self.user_ids or self.default_user_ids() or ["default"])[0]
        try:
            r = self.composio.tools.execute(slug, arguments=call.get("args", {}), user_id=uid, dangerously_skip_version_check=True)
            r = r.model_dump() if hasattr(r, "model_dump") else dict(r)
        except Exception as e:
            return {"ok": False, "error": scrub(f"composio_error: {type(e).__name__}: {str(e)[:200]}")}
        ok = bool(r.get("successful", r.get("successfull", True))) and not r.get("error")
        return {"ok": ok, "data": r.get("data"), **({"error": str(r.get("error"))[:300]} if not ok else {})}
    def _log(self, call, full, kind, status, comp=None, via="call"):
        ev = {"call": call, "full": full, "kind": kind, "status": status, "compact": comp, "via": via}; self.events.append(ev); return ev
    def __call__(self, call):
        """What the agent loop calls. Returns the COMPACT result the model sees. Risky calls are held for approval, not executed."""
        slug = call.get("tool", ""); kind = classify(slug)
        bad = self._check(call)
        if bad: self._log(call, bad, kind, "invalid"); return bad
        if kind == "risky":
            pid = next(self._ids); self.pending.append({"id": pid, "call": call, "source": "chat", "status": "waiting", "t": time.time()})
            res = {"ok": False, "error": "held_for_user_approval", "message": "Not executed yet: waiting for the user to press Approve in the app."}
            self._log(call, res, kind, "held", res); return res
        full = self.run_real(call); comp = compact(full)
        self._log(call, full, kind, "ran", comp); return comp
    def run_step(self, call, source="plan", meta=None):
        """Plan step: read-only -> run now; risky -> held for approval. Returns (status, full, compact)."""
        kind = classify(call.get("tool", "")); bad = self._check(call)
        if bad: self._log(call, bad, kind, "invalid", via="plan"); return "ran", bad, bad
        if kind == "risky":
            self.hold(call, source, meta); res = {"ok": False, "error": "held_for_user_approval"}
            self._log(call, res, kind, "held", res, via="plan"); return "held", None, None
        full = self.run_real(call); comp = compact(full); self._log(call, full, kind, "ran", comp, via="plan"); return "ran", full, comp
    def hold(self, call, source, meta=None):
        pid = next(self._ids); self.pending.append({"id": pid, "call": call, "source": source, "meta": meta, "status": "waiting", "t": time.time()}); return pid
    def approve(self, pid):
        """Run a held call after the user pressed Approve. Returns (full_result, compact_result)."""
        p = next(x for x in self.pending if x["id"] == pid and x["status"] == "waiting")
        full = self.run_real(p["call"]); p["status"] = "approved"; p["result"] = full
        comp = compact(full); self._log(p["call"], full, "risky", "approved-ran", comp, via="approval"); return p, full, comp
    def reject(self, pid):
        p = next(x for x in self.pending if x["id"] == pid and x["status"] == "waiting"); p["status"] = "rejected"; return p
    # ---- watcher counts (read-only) -----------------------------------------------------
    COUNT_ARGS = {"GOOGLEDRIVE_FIND_FILE": {"q": "trashed = false", "pageSize": 1000}, "GMAIL_FETCH_EMAILS": {"max_results": 100}, "GOOGLEDRIVE_FIND_FOLDER": {"page_size": 100}}
    def count(self, slug, polls=1):
        """Real item count for a watcher: runs the watched tool (must be read-only) with minimal args. -> (count|None, note)."""
        if classify(slug) != "read": return None, f"{slug} is not read-only; refusing to poll it"
        c = G.BY.get(slug)
        if c is None: return None, "unknown watch tool"
        args = dict(self.COUNT_ARGS.get(slug, {}))
        miss = [p["name"] for p in c["params"] if p["required"] and p["name"] not in args]
        if miss: return None, f"cannot count with {slug}: needs {', '.join(miss)}"
        full = self.run_real({"tool": slug, "args": args}, small_pages=False); self._log({"tool": slug, "args": args}, full, "read", "watch-poll", compact(full), via="watch")
        if not full["ok"]: return None, scrub(full.get("error"))[:160]
        return count_items(full), "live count of items returned by " + slug + " (capped by the API page size)"


# ---------------------------------------------------------------------------------------------------------------
def to_text(full, max_chars=1500):
    """Readable text of a result, used to fill {{stepN}} references (e.g. a message body made from search results)."""
    if not isinstance(full, dict) or not full.get("ok"): return str(full.get("error") if isinstance(full, dict) else full)[:200]
    data = full.get("data"); lst = [l for _, l in _lists(data) if l and isinstance(l[0], dict)]
    if lst:
        lines = []
        for it in max(lst, key=len)[:15]:
            keys = [k for k in ("subject", "title", "name", "summary", "url", "link", "snippet", "sender", "from") if k in it and not isinstance(it[k], (dict, list))][:3] or [k for k, v in it.items() if not isinstance(v, (dict, list))][:3]
            lines.append("- " + " | ".join(_short(it[k], 90) for k in keys))
        return "\n".join(lines)[:max_chars]
    return json.dumps(data, default=str)[:max_chars]

import runtime2 as R2
OPF = R2.OPF
class LivePlanRunner(R2.PlanRunner):
    """PlanRunner for the app. Runs steps in order; read-only steps run directly, RISKY steps are held for the user's
    Approve button (the plan pauses there); `when` steps fire when tick() sees the condition true (real count in Live mode).
    Plans with a schedule/event trigger are NOT auto-run by this app: use run_now()."""
    def __init__(self, ex):
        super().__init__(ex); self.ex = ex; self.events = []
    def _sub(self, args, outs):
        def f(v):
            if isinstance(v, str):
                for k, o in outs.items(): v = v.replace("{{step%d}}" % k, o)
            return v
        return {k: f(v) for k, v in args.items()}
    def _exec(self, p, i):
        st = p["a"]["steps"][i]; call = {"tool": st["tool"], "args": self._sub(st.get("args", {}), p["outs"])}
        status, full, comp = self.ex.run_step(call, "plan", {"plan": self.plans.index(p), "step": i})
        if status == "held": p["awaiting"] = i; return False
        p["outs"][i + 1] = to_text(full); p["done"].add(i); p.setdefault("results", {})[i] = full
        if not (full or {}).get("ok", True): p["failed"] = i; return False
        return True
    def _advance(self, p, ran=None):
        for i, st in enumerate(p["a"]["steps"]):
            if i in p["done"]: continue
            if p.get("awaiting") is not None or p.get("failed") is not None or p.get("stopped"): break
            if "when" in st: break
            if not self._exec(p, i): break
            if ran is not None: ran.append(i + 1)
    def add(self, a):
        p = {"a": a, "outs": {}, "done": set(), "polls": 0, "awaiting": None}; self.plans.append(p); ran = []
        if not a.get("trigger"): self._advance(p, ran)
        waiting = [i + 1 for i, st in enumerate(a["steps"]) if i not in p["done"]]
        return {"ok": True, "data": {"status": "scheduled", "ran_now": ran, "waiting_steps": waiting, "trigger": a["trigger"]["type"] if a.get("trigger") else None}}
    def run_now(self, p):
        self._advance(p)
    def after_approval(self, pid_entry, full):
        """Called by the UI after an approved plan step ran: records its output and continues the plan."""
        p = self.plans[pid_entry["meta"]["plan"]]; i = pid_entry["meta"]["step"]
        p["outs"][i + 1] = to_text(full); p["done"].add(i); p["awaiting"] = None; p.setdefault("results", {})[i] = full
        if not full.get("ok", True): p["failed"] = i; return
        self._advance(p)
    def after_reject(self, pid_entry):
        self.plans[pid_entry["meta"]["plan"]]["stopped"] = True; self.plans[pid_entry["meta"]["plan"]]["awaiting"] = None
    def tick(self, counts=None):
        """One watcher poll over all plans. Returns a list of human-readable event dicts."""
        out = []
        for pi, p in enumerate(self.plans):
            if p.get("awaiting") is not None or p.get("failed") is not None or p.get("stopped"): continue
            p["polls"] += 1
            for i, st in enumerate(p["a"]["steps"]):
                w = st.get("when")
                if w is None or i in p["done"]: continue
                if not all(j in p["done"] for j in range(i)): break
                n, note = self.ex.count(w["watch"], p["polls"])
                if n is None: out.append({"plan": pi + 1, "step": i + 1, "count": None, "note": note, "fired": False}); break
                hit = OPF[w["op"]](n, w["value"])
                ev = {"plan": pi + 1, "step": i + 1, "watch": w["watch"], "count": n, "cond": f"{w['op']} {w['value']}", "note": note, "fired": hit}
                out.append(ev)
                if hit:
                    ok = self._exec(p, i)
                    ev["held_for_approval"] = p.get("awaiting") == i
                    if ok: self._advance(p)
                break
        self.events += out; return out
