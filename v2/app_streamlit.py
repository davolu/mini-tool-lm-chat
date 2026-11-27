"""Streamlit chat for mini-tool-lm v2 (the from-scratch 13.8M model), with a Mock/Live toggle.

Run:  export COMPOSIO_API_KEY=...   (never put it in a file in the repo)
      streamlit run v2/app_streamlit.py

Live mode uses the Composio accounts that are ALREADY connected (it never creates auth configs or connections):
  * read-only tools (list/get/search/fetch/find...) run directly;
  * everything else (send/post/create/update/upload/delete/forward/move/unknown) is held until you press Approve,
    showing the exact arguments - on top of the model's own confirmation question and the code confirmation gate;
  * the same applies to every risky step of a PLAN when it fires.
The API key is read only from the environment and is never displayed or written."""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); os.chdir(ROOT)
import streamlit as st
import gen as G
import plans as PL; PL.register_extra()
import composio_exec as X
import runtime2 as R2
from runtime import Retriever

# ---- secrets: Streamlit secrets (Cloud "Secrets" box / .streamlit/secrets.toml) first, then real environment variables
for _k in ("COMPOSIO_API_KEY", "COMPOSIO_USER_IDS", "APP_PASSWORD", "HF_TOKEN", "HF_MODEL_REPO"):
    try:
        _v = st.secrets.get(_k)
    except Exception:          # no secrets file at all
        _v = None
    if _v: os.environ[_k] = str(_v)
MODEL_REPO = os.environ.get("HF_MODEL_REPO", "davolu/mini-tool-lm-v2")

st.set_page_config(page_title="mini-tool-lm v2 chat", page_icon="🧩", layout="wide")

# ---- password gate. Fail closed: if a Composio key is configured but no APP_PASSWORD, the app refuses to start
# (set ALLOW_NO_PASSWORD=1 only for local use).
_PW = os.environ.get("APP_PASSWORD")
if not _PW and os.environ.get("COMPOSIO_API_KEY") and os.environ.get("ALLOW_NO_PASSWORD") != "1":
    st.error("APP_PASSWORD is not set. Refusing to start with live Composio access and no password."); st.stop()
if _PW:
    import hmac
    if not st.session_state.get("authed"):
        st.title("🔒 mini-tool-lm v2")
        pw = st.text_input("Password", type="password")
        if pw:
            if hmac.compare_digest(pw.encode(), _PW.encode()): st.session_state.authed = True; st.rerun()
            else: time.sleep(2); st.error("Wrong password.")
        st.stop()

@st.cache_resource(show_spinner="Downloading (first run only) and loading the v2 model and tool retriever...")
def load_models(threads: int):
    local = os.path.join(HERE, "artifacts")
    if os.path.exists(os.path.join(local, "model.pt")) and os.path.exists(os.path.join(local, "tokenizer.json")):
        ckpt, tok = os.path.join(local, "model.pt"), os.path.join(local, "tokenizer.json")
    else:                                   # private HF model repo; cached by huggingface_hub, token from secrets/env
        from huggingface_hub import hf_hub_download
        tk = os.environ.get("HF_TOKEN")
        ckpt = hf_hub_download(MODEL_REPO, "model.pt", token=tk); tok = hf_hub_download(MODEL_REPO, "tokenizer.json", token=tk)
    ret_local = os.path.join(ROOT, "retriever/potion-base-8M")
    try:
        ret = Retriever(path=ret_local if os.path.exists(ret_local) else "minishlab/potion-base-8M")
    except Exception as e:
        st.error("The tool retriever failed to load, so tool search cannot run. " + str(e)[:400])
        raise
    return R2.Brain2(ckpt=ckpt, tok=tok, threads=threads), ret

def pretty(o, n=6000):
    s = json.dumps(o, indent=1, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + f"\n... ({len(s) - n} more characters)"

ss = st.session_state
# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🧩 mini-tool-lm v2")
    has_key = X.LiveExecutor.has_key()
    mode = st.radio("Executor", ["Live", "Mock"], index=0 if has_key else 1,
                    help="Live = real Composio calls on your already-connected accounts. Mock = fake results.")
    if mode == "Live" and not has_key:
        st.error("COMPOSIO_API_KEY is not set in the environment, so Live is unavailable. Falling back to Mock.")
        mode = "Mock"
    uid_text = st.text_input("Composio user id(s)", value=ss.get("uid_text", os.environ.get("COMPOSIO_USER_IDS") or os.environ.get("COMPOSIO_USER_ID") or ""),
                             help="Comma-separated, tried in order per app. Leave blank to auto-detect the id that owns the most connected accounts.") if mode == "Live" else ""
    ss.uid_text = uid_text
    constrained = st.checkbox("Constrained decoding", value=True, help="Tool name must be listed, keys must be that tool's params, values must be spans of the conversation.")
    threads = st.number_input("CPU threads", 1, 16, 4)
    if st.button("🗑 Reset chat"):
        for k in ("agent", "turns", "executor", "mode_active", "uid_active"): ss.pop(k, None)
        st.rerun()

# (re)create agent when the mode / account changes
if ss.get("mode_active") != mode or ss.get("uid_active") != uid_text or "agent" not in ss:
    brain, ret = load_models(int(threads))
    ex = X.LiveExecutor(user_ids=uid_text or None) if mode == "Live" else X.MockExecutor()
    ss.executor = ex; ss.agent = R2.Agent2(brain, ret, executor=ex, constrained=constrained, gate=True)
    ss.agent.plans = X.LivePlanRunner(ex)
    ss.turns = [] ; ss.mode_active = mode; ss.uid_active = uid_text
ag, ex = ss.agent, ss.executor
ag.constrained = constrained

with st.sidebar:
    if mode == "Live":
        try:
            info = ex.connected(); ids = ex.user_ids or ex.default_user_ids()
            st.caption("Account id(s) in use: " + ", ".join(i[:6] + "…" for i in ids[:3]) if ids else "No account ids found")
            st.markdown("**Connected apps** (Composio, existing accounts only)")
            st.write(", ".join(info["connected"]) or "none")
            st.caption("No sign-in needed: " + ", ".join(info["no_auth"]))
            if st.button("↻ Refresh accounts"): ex.refresh(); st.rerun()
        except Exception as e:
            st.error("Could not read Composio accounts: " + X.scrub(str(e))[:200])
        st.info("Live mode: read-only tools run directly; anything that sends / posts / creates / updates / uploads / deletes needs your **Approve** click.")
    else:
        st.info("Mock mode: all results are fake. Nothing touches real apps.")
    st.divider()
    st.markdown("**Plans / watchers**")
    if ag.plans.plans:
        if st.button("👁 Check watchers now", help="Runs one watcher poll: reads the watched tool (live count in Live mode) and fires steps whose condition holds. Risky steps still wait for Approve."):
            ss.last_tick = ag.plans.tick(); st.rerun()
        for ev in ss.get("last_tick", [])[-6:]:
            if ev.get("count") is None: st.warning(f"plan {ev['plan']} step {ev['step']}: {ev['note']}")
            else:
                st.write(f"plan {ev['plan']} step {ev['step']}: count **{ev['count']}** (need {ev['cond']}) → " + ("**condition met**" + (" - waiting for your Approve" if ev.get("held_for_approval") else " - ran") if ev["fired"] else "not yet"))
                st.caption(ev["note"])
    else:
        st.caption("No plan yet. Try a long request, e.g. “search for latest news and save it in my Google drive, then forward it to ada@x.com once the items in Google drive are more than 20”.")

# ---------------------------------------------------------------- rendering helpers
STATUS_ICON = {"done": "✅", "awaiting": "⏸️ waiting for your Approve", "watch": "👁 watching", "pending": "⏳ pending", "failed": "❌ failed", "stopped": "🛑 stopped"}
def plan_card(pi):
    p = ag.plans.plans[pi]; a = p["a"]
    st.markdown(f"**Plan {pi + 1}**" + (f" - trigger: `{a['trigger']['type']}` {a['trigger'].get('every') or a['trigger'].get('match')} (not auto-run by this app)" if a.get("trigger") else ""))
    for i, s in enumerate(a["steps"]):
        c = G.BY.get(s["tool"]); risky = X.classify(s["tool"]) == "risky"
        if i in p["done"]: stt = "done"
        elif p.get("failed") == i: stt = "failed"
        elif p.get("stopped"): stt = "stopped"
        elif p.get("awaiting") == i: stt = "awaiting"
        elif "when" in s: stt = "watch"
        else: stt = "pending"
        w = s.get("when")
        line = f"{i + 1}. `{s['tool']}`" + (" 🔒risky" if risky else "") + f" - {STATUS_ICON[stt]}"
        if w: line += f" · **when** count(`{w['watch']}`) {w['op']} {w['value']}"
        st.markdown(line)
        with st.expander(f"step {i + 1} args" + (" / result" if i in p.get("results", {}) else ""), expanded=False):
            st.code(pretty(s.get("args", {})), language="json")
            if i in p.get("results", {}): st.code(pretty(p["results"][i]), language="json")
    if a.get("trigger") and st.button("▶ Run this plan now (manual)", key=f"run{pi}"):
        ag.plans.run_now(p); st.rerun()

def render_turn(t):
    if "note" in t:
        with st.chat_message("assistant", avatar="🛡️"): st.markdown(t["note"]); return
    with st.chat_message("user"): st.write(t["user"])
    with st.chat_message("assistant"):
        if t.get("tools"):
            with st.expander(f"🔎 retrieved tools ({len(t['tools'])})"):
                for n in t["tools"]:
                    c = G.BY.get(n); st.markdown(f"- `{n}` {'🔒' if X.classify(n) == 'risky' else '👁'} {c['desc'][:110] if c else ''}")
        evs = [e for e in t["events"] if e.get("via") == "call"]; ei = 0; pl = list(t.get("plan_ids", []))
        for role, txt in t["trace"][1:]:
            if role == "think": st.caption("💭 " + txt)
            elif role == "call":
                try: d = json.loads(txt)
                except Exception: d = None
                if d and d.get("tool") == "PLAN":
                    st.markdown("📋 **PLAN written by the model**")
                    if pl: plan_card(pl.pop(0))
                    with st.expander("plan JSON"): st.code(pretty(d), language="json")
                else:
                    risky = d and X.classify(d.get("tool", "")) == "risky"
                    with st.expander(f"🔧 call: {d.get('tool') if d else '(invalid)'}" + (" 🔒" if risky else ""), expanded=False): st.code(pretty(d) if d else txt, language="json")
            elif role == "result":
                e = evs[ei] if ei < len(evs) else None; ei += 1
                label = "📥 result" + (f" ({e['status']})" if e else "")
                if e and e["full"].get("ok") and e.get("status") in ("ran", "mock") and X.count_items(e["full"]):
                    st.markdown("**Actual results** (straight from the tool; the model's summary below can be inaccurate):")
                    st.markdown(X.to_text(e["full"], 900))
                with st.expander(label, expanded=False):
                    if e: st.caption("what the model saw:"); st.code(pretty(e.get("compact") or e["full"], 1500), language="json"); st.caption("full result:"); st.code(pretty(e["full"]), language="json")
                    else: st.code(txt, language="json")
            elif role == "assistant":
                st.markdown(("🔐 " if txt.startswith("[gate]") else "") + txt)

# ---------------------------------------------------------------- main
st.title("Chat with mini-tool-lm v2")
st.caption(f"Mode: **{mode}** · 13.8M-parameter model trained from scratch · results can be wrong: check every approval carefully.")
for t in ss.turns: render_turn(t)

# approvals (Live): exact args, Approve / Reject
waiting = [p for p in getattr(ex, "pending", []) if p["status"] == "waiting"]
for p in waiting:
    with st.container(border=True):
        c = p["call"]; tk = X.toolkit_of(c["tool"])
        where = "plan step %d" % (p["meta"]["step"] + 1) if p.get("meta") else "chat"
        st.warning(f"**Approval needed** - `{c['tool']}` on **{tk}** ({where}). Exact arguments:")
        st.code(pretty(c.get("args", {})), language="json")
        b1, b2, _ = st.columns([1, 1, 4])
        if b1.button("✅ Approve & run", key=f"ok{p['id']}", type="primary"):
            with st.spinner("running on Composio..."):
                ent, full, comp = ex.approve(p["id"])
            if p.get("source") == "plan": ag.plans.after_approval(p, full)
            ag.history.append(["result", json.dumps(comp)])
            msg = f"✅ Approved and ran `{c['tool']}`: " + ("worked." if full.get("ok") else "FAILED - " + str(full.get("error") or full.get("message"))[:200])
            ag.history.append(["assistant", msg.replace("`", "")])
            ss.turns.append({"note": msg}); st.rerun()
        if b2.button("✖ Reject", key=f"no{p['id']}"):
            ex.reject(p["id"])
            if p.get("source") == "plan": ag.plans.after_reject(p)
            msg = f"✖ Rejected `{c['tool']}` - nothing was executed."
            ag.history.append(["assistant", "Okay, I won't do it."]); ss.turns.append({"note": msg}); st.rerun()

prompt = st.chat_input("Ask something, e.g. “find emails about invoice” or “search hacker news for AI agents”")
if prompt:
    n0, e0, p0 = len(ag.trace), len(ex.events), len(ag.plans.plans)
    with st.spinner("thinking..."):
        try: ag.send(prompt)
        except Exception as e: ag.trace.append(["assistant", "(error: " + X.scrub(str(e))[:200] + ")"])
    new_events = list(ex.events[e0:])
    held = [e for e in new_events if e["status"] == "held"]
    if held and ag.history and ag.history[-1][0] in ("result", "call"):
        ag.history.append(["assistant", "Waiting for your approval in the app."])   # the model was never trained on a 'held' result
        ag.trace.append(["assistant", "⏸ " + ", ".join(sorted({e["call"]["tool"] for e in held})) + " is waiting for your **Approve / Reject** below - nothing has been executed yet."])
    ss.turns.append({"user": prompt, "trace": list(ag.trace[n0:]), "events": new_events, "plan_ids": list(range(p0, len(ag.plans.plans))),
                     "tools": [c["name"] for c in (getattr(ag, "_last_tools", None) or [])]})
    st.rerun()
