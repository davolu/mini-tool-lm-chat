"""Long, natural, multi-step and conditional requests -> a structured PLAN that ordinary code executes and watches.
Model output:  <|call|>{"tool": "PLAN", "args": {"steps": [{"tool": T, "args": {...}, "when": {...}?}, ...], "trigger": {...}?}}<|eot|>
  step args may reference earlier outputs as "{{step1}}";  when = {"type": "count", "watch": LIST_TOOL, "op": ">", "value": 20}
  trigger = {"type": "schedule", "every": "Monday at 9am"}  or  {"type": "event", "watch": TOOL, "match": "boss@x.com"}
Vague references ("my main", "my boss") -> ask first. Plans with risky steps -> confirm first."""
import random, json, re
import gen as G

EXTRA = [{"name": "GOOGLEDRIVE_CREATE_FILE_FROM_TEXT", "app": "googledrive", "category": "files", "split": "train",
          "desc": "Creates a new file in Google Drive from text content", "returns": [], "risky": False,
          "params": [{"name": "file_name", "type": "str", "required": True, "desc": "file name"},
                     {"name": "text_content", "type": "str", "required": True, "desc": "text content"},
                     {"name": "parent_id", "type": "str", "required": False, "desc": "folder id"}]}]
def register_extra():
    for c in EXTRA:
        if c["name"] not in G.BY:
            G.BY[c["name"]] = c; G.CAT.append(c); G.APP_TOOLS[c["app"]].append(c)

def _req(c): return [p for p in c["params"] if p["required"]]
def roles(pool):
    R = {"source": [], "store": [], "action": [], "watch": {}}
    for c in pool:
        vc = G.verb_class(c); sems = {psem(p) for p in c["params"]}; rs = {psem(p) for p in _req(c)}
        if c["app"] in ("openweather_api", "github"): continue
        if vc == "read" and "query" in sems and rs <= {"query"}: R["source"].append(c)
        if vc == "read" and rs <= {"query"}: R["watch"].setdefault(c["app"], []).append(c)
        if vc == "create" and ("body" in sems or "title" in sems) and rs <= {"body", "title", "path"} and c["category"] in ("files", "docs", "notes", "tasks"):
            R["store"].append(c)
        if vc == "send" and "body" in sems and rs <= {"email", "channel", "body", "title"} and (rs & {"email", "channel"} or c["category"] == "social"):
            R["action"].append(c)
    R["store"] = [c for c in R["store"] if R["watch"].get(c["app"])]
    return R

VAGUE = {"email": ["my main", "my main email", "my boss", "my other account", "my assistant", "my work email", "the team"],
         "channel": ["the usual channel", "our team channel", "the group chat"]}
OPS = [(">", ["more than {n}", "over {n}", "above {n}", "greater than {n}"]), (">=", ["at least {n}", "{n} or more"]), ("<", ["fewer than {n}", "less than {n}", "below {n}"])]
DAYS = ["Monday at 9am", "morning at 8am", "day at 6pm", "Friday at 5pm", "Sunday evening", "hour", "weekday at 7am", "1st of the month"]

def psem(p):
    n = p["name"].lower()
    if re.fullmatch(r"(file_?name|name|title|doc_?name|subject)", n): return "title"
    if re.fullmatch(r"(text_content|content|text|body|markdown_text|message|note)", n): return "body"
    return G.sem(p)
def pick(c, sem_set):
    return next((p for p in c["params"] if psem(p) in sem_set), None)

def app(c): return G.APPNAME[c["app"]]

def source_clause(c, q):
    if c["app"] == "composio_search":
        return random.choice([f"search for {q}", f"find the latest {q}", f"look up {q} online", f"grab the newest {q}", f"pull the latest news on {q}", f"search the web for {q}"])
    return random.choice([f"search {app(c)} for {q}", f"find {G.object_of(c)} about {q} on {app(c)}", f"pull {q} from {app(c)}", f"look up {q} in {app(c)}", f"get my {G.object_of(c)} about {q} from {app(c)}"])

def store_clause(c, title):
    t = f" as {title}" if title else ""
    return random.choice([f"save them to my {app(c)}{t}", f"populate them in my {app(c)}{t}", f"put the results in {app(c)}{t}", f"store everything in {app(c)}{t}",
                          f"add them to {app(c)}{t}", f"drop them into my {app(c)}{t}", f"create a {G.object_of(c)} in {app(c)}{' called ' + title if title else ''} with the results"])

def action_clause(c, who):
    sem_who = G.sem(pick(c, {"email", "channel"})) if pick(c, {"email", "channel"}) else None
    if c["category"] == "social" and not sem_who:
        return random.choice([f"post a summary on {app(c)}", f"share it on {app(c)}", f"publish it on my {app(c)}"])
    return random.choice([f"forward it to {who}", f"send it to {who}", f"email the link to {who}", f"share it with {who} on {app(c)}", f"send {who} a message on {app(c)} with it",
                          f"notify {who} via {app(c)}"]) if sem_who == "email" else random.choice([f"post it in {who}", f"send it to {who} on {app(c)}", f"drop a message in {who}", f"let {who} know on {app(c)}"])

FILL = ["", "", "", "hey so here's what I need: ", "ok listen, ", "I want to automate something: ", "can you set this up for me - ", "here's the workflow: ", "quick automation: ",
        "so basically ", "I need a small routine: ", "please handle this end to end: "]
END = ["", "", ".", " thanks", " please", ". can you do that?", ". make sure it works", " and let me know when it's set up"]

def make_plan_conv(pool, R=None, force_vague=None, real_names_only=False):
    R = R or roles(pool)
    src = random.choice(R["source"]); nsteps = random.choice([2, 3, 3, 3])
    q = G.value_for(pick(src, {"query"}), src)
    steps, clauses, slots = [], [], [q]
    qp = pick(src, {"query"}); steps.append({"tool": src["name"], "args": {qp["name"]: q}}); clauses.append(source_clause(src, q))
    store = None
    if nsteps == 3 or random.random() < 0.5:
        store = random.choice(R["store"]); title = None
        tp = pick(store, {"title"}); bp = pick(store, {"body"})
        a = {}
        if tp: title = G.vstr(G.value_for(tp, store)); a[tp["name"]] = title; slots.append(title)
        if bp: a[bp["name"]] = "{{step1}}"
        for p in _req(store):
            if p["name"] not in a: a[p["name"]] = "{{step1}}" if psem(p) != "title" else (title or "results")
        steps.append({"tool": store["name"], "args": a}); clauses.append(store_clause(store, title if tp and random.random() < .7 else None))
        if not (tp and title in clauses[-1]) and tp: a[tp["name"]] = title  # still copied: title must appear in text
        if tp and title not in clauses[-1]: clauses[-1] += f" named {title}"
    act = None
    if nsteps == 3 or store is None:
        act = random.choice(R["action"]); a = {}
        wp = pick(act, {"email", "channel"}); bp = pick(act, {"body"})
        vague = None
        if wp:
            who = G.vstr(G.value_for(wp, act)); a[wp["name"]] = [who] if wp["type"] == "list" else who
            fv = force_vague if force_vague is not None else (random.random() < 0.3)
            if fv: vague = random.choice(VAGUE["email" if G.sem(wp) == "email" else "channel"])
        else: who = None
        if bp: a[bp["name"]] = "{{step%d}}" % len(steps)
        for p in _req(act):
            if p["name"] not in a: a[p["name"]] = "{{step%d}}" % len(steps)
        steps.append({"tool": act["name"], "args": a}); clauses.append(action_clause(act, vague or who))
        if who and not vague: slots.append(who)
    # condition / trigger
    trigger = None; cond_text = None; kind = random.choice(["count", "count", "schedule", "event", "none"])
    if kind == "count" and store is not None and act is not None:
        w = random.choice(R["watch"][store["app"]]); op, phr = random.choice(OPS); n = random.choice([5, 10, 15, 20, 25, 30, 50, 100])
        steps[-1]["when"] = {"type": "count", "watch": w["name"], "op": op, "value": n}
        cond_text = random.choice([f"once the items in {app(store)} are ", f"when {app(store)} has ", f"as soon as there are ", f"only when the count in {app(store)} is "]) + random.choice(phr).format(n=n) + random.choice(["", " items", " files", " entries"])
    elif kind == "schedule":
        d = random.choice(DAYS); trigger = {"type": "schedule", "every": d}; cond_text = f"every {d}"
    elif kind == "event":
        ew = [c for c in pool if c["name"] in ("GMAIL_FETCH_EMAILS", "SLACK_SEARCH_MESSAGES")] or R["source"][:1]
        e = random.choice(ew); m = G.email() if "GMAIL" in e["name"] else random.choice(G.CHANNELS)
        trigger = {"type": "event", "watch": e["name"], "match": m}; cond_text = (f"whenever I get an email from {m}" if "GMAIL" in e["name"] else f"whenever there's a new message in {m}"); slots.append(m)
        if e["name"] not in [s["tool"] for s in steps]: pass
    # phrasing
    conn = [random.choice([", then ", " and then ", ". After that ", ", and after that ", " then "]) for _ in range(len(clauses))]
    if cond_text and "when" in steps[-1]:
        last = clauses[-1]
        if random.random() < 0.5: body = clauses[0] + "".join(conn[i] + clauses[i] for i in range(1, len(clauses) - 1)) + conn[-1] + last + " " + cond_text
        else: body = clauses[0] + "".join(conn[i] + clauses[i] for i in range(1, len(clauses) - 1)) + random.choice([". And ", ", and ", ". "]) + cond_text + ", " + last
    else:
        body = clauses[0] + "".join(conn[i] + clauses[i] for i in range(1, len(clauses)))
        if cond_text: body = (cond_text + ", " + body) if random.random() < 0.5 else (body + " " + cond_text)
    if random.random() < 0.3: body = "first " + body
    text = G.style(random.choice(FILL) + body + random.choice(END), slots)
    plan = {"steps": steps}
    if trigger: plan["trigger"] = trigger
    tools_needed = [G.BY[s["tool"]] if s["tool"] in G.BY else next(c for c in pool if c["name"] == s["tool"]) for s in steps]
    for extra in ([steps[-1]["when"]["watch"]] if "when" in steps[-1] else []) + ([trigger["watch"]] if trigger and "watch" in trigger else []):
        c = G.BY.get(extra) or next(c for c in pool if c["name"] == extra)
        if c not in tools_needed: tools_needed.append(c)
    k = max(0, random.randint(5, 7) - len(tools_needed))
    others = [c for c in pool if c not in tools_needed]
    tools = tools_needed + random.sample(others, k); random.shuffle(tools)
    segs = [["tools", G.tools_block(tools)], ["user", text]]
    desc = " | ".join(f"{i+1}: {s['tool']}" + (f" when count({s['when']['watch']}) {s['when']['op']} {s['when']['value']}" if "when" in s else "") for i, s in enumerate(steps))
    if trigger: desc += f" | trigger: {trigger['type']} {trigger.get('every', trigger.get('match', ''))}"
    vague_ref = None
    if act is not None and pick(act, {"email", "channel"}) and any(v in text for vs in VAGUE.values() for v in vs):
        vague_ref = next(v for vs in VAGUE.values() for v in sorted(vs, key=len, reverse=True) if v in text)
    risky = any((G.BY.get(s["tool"]) or next(c for c in pool if c["name"] == s["tool"]))["risky"] for s in steps)
    meta = {"type": "plan", "plan": plan, "vague": vague_ref, "risky": risky}
    if vague_ref:
        segs.append(["think", f"multi-step plan | vague reference: '{vague_ref}' -> ask"])
        wp = pick(act, {"email", "channel"})
        q = (f"Who is '{vague_ref}'? What's the email address?" if G.sem(wp) == "email" else f"Which channel is '{vague_ref}'?")
        segs.append(["assistant", random.choice([q, f"Quick question before I set this up: {q}", f"Got it. {q}"])])
        val = steps[-1]["args"][wp["name"]]; val = val[0] if isinstance(val, list) else val
        segs.append(["user", random.choice([val, f"it's {val}", f"{val} please", f"use {val}"])])
        segs.append(["think", f"got {wp['name']} | plan: {desc}" + (" | confirm first" if risky else "")])
    else:
        segs.append(["think", f"multi-step plan: {desc}" + (" | confirm first" if risky else "")])
    if risky:
        lines = []
        for i, s in enumerate(steps):
            c = G.BY.get(s["tool"]) or next(c for c in pool if c["name"] == s["tool"])
            ln = f"{i+1}) {G.verb_class(c)} {G.object_of(c)} on {app(c)}"
            if "when" in s: ln += f" - only when {G.object_of(G.BY.get(s['when']['watch']) or next(c for c in pool if c['name'] == s['when']['watch']))} count {s['when']['op']} {s['when']['value']}"
            lines.append(ln)
        tr = f" It will run {('every ' + trigger['every']) if trigger and trigger['type']=='schedule' else ('whenever ' + trigger['match'] + ' shows up') if trigger else 'now'}."
        segs.append(["assistant", "Here's the plan: " + "; ".join(lines) + "." + tr + random.choice([" Shall I set it up?", " Okay to go ahead?", " Confirm?"])])
        segs.append(["user", random.choice(G.YES)])
        segs.append(["think", "confirmed -> PLAN"])
    utext = " ".join(t for r, t in segs if r == "user")
    def fix(v):
        if isinstance(v, list): return [fix(x) for x in v]
        if not isinstance(v, str) or v.startswith("{{") or v in utext: return v
        i = utext.lower().find(v.lower())
        return utext[i:i + len(v)] if i >= 0 else v
    for st in steps: st["args"] = {k: fix(v) for k, v in st["args"].items()}
    if trigger and "match" in trigger: trigger["match"] = fix(trigger["match"])
    segs.append(["call", json.dumps({"tool": "PLAN", "args": plan}, ensure_ascii=False)])
    waiting = [i + 1 for i, s in enumerate(steps) if "when" in s]
    ran = [] if trigger else [i + 1 for i, s in enumerate(steps) if "when" not in s]
    res = {"ok": True, "data": {"status": "scheduled", "ran_now": ran, "waiting_steps": waiting, "trigger": trigger["type"] if trigger else None}}
    segs.append(["result", json.dumps(res)])
    if trigger: rep = f"All set. The plan will run {('every ' + trigger['every']) if trigger['type']=='schedule' else 'whenever ' + trigger['match'] + ' shows up'}."
    elif waiting:
        w = steps[-1]["when"]; rep = f"Done - I ran steps {', '.join(map(str, ran))} and I'm watching it: step {waiting[0]} will run when the count is {w['op']} {w['value']}."
    else: rep = f"Done - all {len(steps)} steps ran."
    segs.append(["assistant", rep])
    return segs, meta, tools
