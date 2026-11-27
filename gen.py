"""Synthetic conversation generator for the mini tool-calling model.
Each conversation = list of [role, text] segments; roles: tools,user,think,call,result,assistant.
Model is trained only on think/call/assistant segments (+ the special token that opens them).
"""
import json, random, re, sys
from collections import defaultdict

CAT = json.load(open("data/catalog.json"))
BY = {c["name"]: c for c in CAT}
APPNAME = {"gmail": "Gmail", "outlook": "Outlook", "googlecalendar": "Google Calendar", "calendly": "Calendly", "zoom": "Zoom",
           "twitter": "Twitter", "linkedin": "LinkedIn", "facebook": "Facebook", "instagram": "Instagram", "youtube": "YouTube",
           "reddit": "Reddit", "slack": "Slack", "discord": "Discord", "telegram": "Telegram", "googledrive": "Google Drive",
           "dropbox": "Dropbox", "googledocs": "Google Docs", "notion": "Notion", "googlesheets": "Google Sheets", "airtable": "Airtable",
           "asana": "Asana", "trello": "Trello", "jira": "Jira", "github": "GitHub", "composio_search": "web search",
           "hackernews": "Hacker News", "openweather_api": "OpenWeather", "hubspot": "HubSpot", "stripe": "Stripe"}
APPWORDS = {"NOTION", "LINKED", "IN", "YOU", "TUBE", "GOOGLE", "SLACK", "IG", "BY", "ID", "A", "AN", "THE", "OF", "FOR", "TO", "ME", "MY", "ALL", "USER", "CURRENT"}

# ----------------------------------------------------------------- tool semantics
def action_of(c):
    n, app = c["name"], c["app"].upper()
    if n.startswith(app + "_"): return n[len(app) + 1:]
    return n.split("_", 1)[1]

VERBS = {"send": ["SEND", "REPLY", "FORWARD", "POST", "CREATION", "PUBLISH", "SHARE", "INVITE"],
         "read": ["LIST", "FETCH", "GET", "RETRIEVE", "SEARCH", "FIND", "QUERY", "RECENT", "LOOKUP", "WEB", "NEWS", "READ", "DESCRIBE", "VIEW"],
         "create": ["CREATE", "ADD", "INSERT", "QUICK", "UPLOAD", "NEW", "MAKE", "SCHEDULE", "COPY", "APPEND", "START"],
         "delete": ["DELETE", "REMOVE", "TRASH", "ARCHIVE", "MOVE_TO_TRASH", "CANCEL", "BATCH_DELETE"],
         "update": ["UPDATE", "PATCH", "MODIFY", "EDIT", "SET", "MOVE", "RENAME", "MARK", "LIKE", "STAR", "PIN"]}

def verb_class(c):
    a = action_of(c)
    if c["app"] == "composio_search": return "read"
    for cls, vs in VERBS.items():
        for v in vs:
            if a.startswith(v): return cls
    return "read"

def object_of(c):
    a = action_of(c)
    w = [x for x in a.split("_") if x not in APPWORDS]
    first = w[0] if w else ""
    if any(first.startswith(v) for vs in VERBS.values() for v in vs) or first in ("CREATION",):
        w = w[1:]
    obj = " ".join(x.lower() for x in w).strip()
    if c["app"] == "composio_search": obj = (obj + " results").strip()
    return obj or {"email": "emails", "social": "posts", "chat": "messages", "files": "files", "calendar": "events"}.get(c["category"], "items")

VERB_WORDS = {"send": ["send", "post", "send out"], "read": ["show me", "get", "list", "find", "pull up", "fetch", "look up", "check"],
              "create": ["create", "make", "add", "set up", "create a new"], "delete": ["delete", "remove", "get rid of", "trash"],
              "update": ["update", "change", "edit", "modify"]}
PAST = {"send": "sent", "read": "found", "create": "created", "delete": "deleted", "update": "updated"}

def sem(p, c=None):
    n, t = p["name"], p["type"]; nl = n.lower()
    if t == "bool": return "bool"
    if t in ("int", "num") and nl != "page" and any(k in nl for k in ("limit", "max", "top", "count", "size", "results")): return "count"
    if t in ("int", "num"): return "number"
    if "email" in nl or nl in ("to", "recipient", "to_recipients", "attendees"): return "email"
    if nl in ("channel", "chat_id", "channel_id", "conversation"): return "channel"
    if nl in ("body", "text", "message", "message_body", "markdown_text", "commentary", "content", "comment", "description", "desc",
              "notes", "hs_task_body", "markdown", "caption", "status", "tweet_text", "comment_text"): return "body"
    if nl in ("subject", "summary", "title", "name", "hs_task_subject", "folder_name", "label_name", "topic"): return "title"
    if nl in ("query", "q", "search", "keyword", "keywords", "term", "subject_contains"): return "query"
    if any(k in nl for k in ("time", "date", "when", "due", "timestamp", "datetime", "since", "until")): return "time"
    if nl in ("location", "city", "place", "address", "region", "country"): return "location"
    if nl in ("path", "folder", "folder_path", "parent", "destination"): return "path"
    if nl.endswith("id") or nl.endswith("_ids") or nl.endswith("ts") or nl in ("code", "key", "repo", "owner", "project_key", "idlist", "author", "username", "handle"):
        return "id"
    return "generic"

def pwords(n):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", n).replace("_", " ").lower()
    return s.replace("hs ", "").strip()

# ----------------------------------------------------------------- value pools
FIRST = ["ada", "tunde", "chioma", "bola", "john", "sarah", "mike", "fatima", "emeka", "grace", "kemi", "david", "lisa", "omar", "ngozi",
         "peter", "yemi", "amara", "kofi", "zainab", "tom", "anna", "seun", "musa", "ifeoma", "james", "maria", "uche", "dami", "nora"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.com", "acme.io", "startup.ng", "mail.com", "work.org"]
SUBJ = ["Project update", "Invoice for November", "Meeting tomorrow", "Quick question", "Weekly report", "Contract draft", "Team lunch",
        "Budget review", "Launch plan", "Client feedback", "Holiday schedule", "Q4 targets", "Design review", "New hire onboarding",
        "Sprint planning", "Church outreach", "Youtube script", "Product demo", "Rent receipt", "Partnership proposal", "Website redesign"]
BODY_A = ["I'll be 10 minutes late", "the report is ready", "can we move the meeting to Friday", "thanks for your help today",
          "please send me the invoice", "the launch is on track", "I've shared the files with you", "let's catch up next week",
          "happy birthday", "the client approved the design", "remember to bring your laptop", "the deadline is Monday",
          "I won't make it to the call", "new video is live", "great job on the demo", "can you review the draft",
          "dinner is at 7", "we hit our target this month", "the server is back up", "please confirm the booking"]
BODY_B = ["", "", "", " and I'll follow up later", ", talk soon", ". Thanks!", " and let me know if you have questions", ", cheers"]
TOPICS = ["AI agents", "the budget", "invoices", "the marketing plan", "flights to London", "react hooks", "python tutorials", "Nigeria tech news",
          "the quarterly report", "climate change", "electric cars", "the product launch", "job openings", "bitcoin price", "the new iPhone",
          "comedy skits", "football scores", "healthy recipes", "remote work", "the contract", "Lagos traffic", "startup funding"]
TITLES = ["Team sync", "Dentist appointment", "Launch checklist", "Meeting notes", "Q4 plan", "Grocery list", "Video ideas", "Client call",
          "Bug triage", "Content calendar", "Gym session", "Budget 2027", "Interview with Sarah", "Weekly standup", "Book club", "Sermon notes",
          "Fix login bug", "Design sprint", "Marketing ideas", "Travel plans", "Onboarding doc", "Release v2"]
TIMES = ["tomorrow at 3pm", "next Monday at 10am", "Friday at noon", "today at 5pm", "2026-12-05 14:00", "tonight at 8", "Dec 12 at 9am",
         "next week Tuesday 11am", "in 2 hours", "Saturday morning", "2026-12-01T09:00:00", "this Thursday 4pm", "Jan 3 at 2pm"]
CITIES = ["Lagos", "Abuja", "London", "New York", "Accra", "Nairobi", "Paris", "Toronto", "Dubai", "Ibadan", "Port Harcourt", "Berlin"]
CHANNELS = ["#general", "#marketing", "#dev", "#random", "#sales", "#team", "#announcements", "#design", "#support"]
PATHS = ["/Projects", "/Invoices/2026", "/Videos", "/Documents/Contracts", "/Photos", "/Work/Reports", "/Shared"]
GENERIC = ["high", "public", "weekly", "draft", "blue", "en", "default", "primary", "urgent", "Task", "Bug", "low", "private", "monthly"]

def rid(n=None):
    return "".join(random.choice("0123456789abcdef") for _ in range(n or random.randint(8, 14)))

def email():
    f = random.choice(FIRST)
    return random.choice([f, f + "." + random.choice(FIRST), f + str(random.randint(1, 99))]) + "@" + random.choice(DOMAINS)

def body():
    return random.choice(BODY_A) + random.choice(BODY_B)

def value_for(p, c=None):
    s, n = sem(p), p["name"].lower()
    if s == "email": v = email(); return [v] if p["type"] == "list" else v
    if s == "count": return random.choice([3, 5, 5, 10, 10, 20, 2, 7, 15])
    if s == "bool": return True
    if s == "channel": return random.choice(CHANNELS) if n != "chat_id" else random.choice(["@" + random.choice(FIRST) + "_chat", str(-1000000 - random.randint(1, 999999))])
    if s == "body": return body()
    if s == "title": return random.choice(SUBJ if n == "subject" else TITLES)
    if s == "query":
        if c and c["app"] == "openweather_api": return random.choice(CITIES)
        return random.choice(TOPICS)
    if s == "time": return random.choice(TIMES)
    if s == "location": return random.choice(CITIES)
    if s == "path": return random.choice(PATHS)
    if s == "id":
        if n == "repo": return random.choice(["website", "mini-jev", "api-server", "docs"])
        if n == "owner": return random.choice(FIRST)
        if n == "project_key": return random.choice(["WEB", "APP", "OPS", "MKT"])
        return rid()
    if s == "number":
        return round(random.uniform(-90, 90), 4) if ("lat" in n or "lon" in n or p["type"] == "num") else random.randint(1, 50)
    v = random.choice(GENERIC)
    return [v] if p["type"] == "list" else v

def vstr(v):
    return ", ".join(map(str, v)) if isinstance(v, list) else ("on" if v is True else str(v))

# ----------------------------------------------------------------- mention templates (generic, work for any tool)
def mention(p, v, c):
    s, pw, obj = sem(p), pwords(p["name"]), object_of(c)
    vs = vstr(v); q = random.random() < 0.4
    qv = f'"{vs}"' if q else vs
    n = p["name"].lower()
    T = {
        "email": [f"to {vs}", f"to {vs}", f"for {vs}", f"with {pw} {vs}"],
        "body": [f"saying {qv}", f"that says {qv}", f"with the message: {vs}", f'with text "{vs}"', f"with {pw} {qv}", f"and tell them {vs}"],
        "title": [f"titled {qv}", f"called {qv}", f"named {qv}", f"with {pw} {qv}"] + ([f"with subject {qv}", f"subject {qv}"] if n == "subject" else []),
        "query": [f"about {vs}", f"for {vs}", f'matching "{vs}"', f"on {vs}", f"with {pw} {qv}"],
        "channel": [f"in {vs}", f"to {vs}", f"in {pw} {vs}"],
        "time": ([f"until {vs}", f"ending {vs}", f"with {pw} {vs}"] if any(k in n for k in ("end", "max", "until")) else [f"{vs}", f"for {vs}", f"starting {vs}", f"with {pw} {vs}"]),
        "location": [f"in {vs}", f"at {vs}", f"with {pw} {vs}"],
        "path": [f"in {vs}", f"at {vs}", f"with {pw} {vs}"],
        "id": [f"{pw} {vs}", f"with {pw} {vs}", f"{pw}: {vs}", f"(id {vs})" if pw.endswith("id") else f"{pw} {vs}"],
        "count": [f"limit {vs}", f"just {vs} of them", f"max {vs}", f"{vs} results", f"with {pw} {vs}"],
        "bool": [f"with {pw} on", f"and set {pw} to true", f"{pw} = true"],
        "number": [f"with {pw} {vs}", f"{pw} {vs}", f"{pw}: {vs}", f"and set {pw} to {vs}"],
        "generic": [f"with {pw} {qv}", f"{pw}: {vs}", f"and set {pw} to {qv}", f"{pw} {qv}"],
    }[s]
    return random.choice(T)

def app_phrase(c):
    a = APPNAME[c["app"]]
    if c["app"] == "composio_search": return random.choice(["", "online", "on the web"])
    return random.choice([f"in {a}", f"on {a}", f"in {a}", f"using {a}", f"on my {a}", f"via {a}", ""])

def generic_request(c, params_to_mention, vals):
    vc, obj = verb_class(c), object_of(c)
    sv = action_of(c).split("_")[0].lower()
    if c["app"] == "composio_search": sv = "search"
    canon = sv in ("list", "get", "fetch", "search", "find", "create", "add", "send", "delete", "update", "remove", "retrieve")
    verb = random.choice(VERB_WORDS[vc]) if canon and random.random() < 0.5 else sv
    det = random.choice(["a ", "the ", "my ", ""]) if vc != "read" else random.choice(["", "my ", "the "])
    parts = [m for p in params_to_mention for m in [mention(p, vals[p["name"]], c)]]
    random.shuffle(parts)
    core = f"{verb} {det}{obj} {app_phrase(c)}".replace("  ", " ").strip()
    return (core + " " + " ".join(parts)).strip()

# ----------------------------------------------------------------- natural templates for common tools
NAT = {
 "GMAIL_FETCH_EMAILS": ["check my mail", "any new emails?", "show me my latest emails", "check my inbox", "what's in my inbox",
    "check my inbox for emails about {query}", "find emails about {query}", "show me the last {max_results} emails", "did i get any mail about {query}",
    "read my emails", "show my last {max_results} emails about {query}", "do I have new mail"],
 "GMAIL_SEND_EMAIL": ["email {recipient_email} and tell them {body}", "send an email to {recipient_email} saying {body}",
    "send a mail to {recipient_email} with subject {subject} saying {body}", "shoot {recipient_email} an email: {body}",
    "send an email to {recipient_email}", "send an email saying {body}", "i need to send an email", "email {recipient_email}",
    "write to {recipient_email} that {body}", "mail {recipient_email} \"{body}\""],
 "GMAIL_CREATE_EMAIL_DRAFT": ["draft an email to {recipient_email} saying {body}", "create a draft for {recipient_email}: {body}",
    "draft a mail to {recipient_email}", "prepare an email draft saying {body}", "write a draft email to {recipient_email} about {subject} saying {body}"],
 "GMAIL_SEARCH_PEOPLE": ["find {query} in my contacts", "look up {query}'s contact", "search my contacts for {query}", "what's {query}'s email"],
 "GMAIL_LIST_LABELS": ["list my gmail labels", "what labels do I have", "show my email labels"],
 "GMAIL_MOVE_TO_TRASH": ["trash email {message_id}", "move message {message_id} to trash", "delete the email {message_id}", "trash that email"],
 "GOOGLECALENDAR_CREATE_EVENT": ["schedule {summary} {start_datetime}", "add {summary} to my calendar for {start_datetime}",
    "create an event called {summary} {start_datetime}", "put {summary} on my calendar {start_datetime} at {location}",
    "book {summary} on my calendar", "add an event {start_datetime}", "create a calendar event", "set up a meeting {start_datetime}",
    "schedule {summary} {start_datetime} till {end_datetime}"],
 "GOOGLECALENDAR_EVENTS_LIST": ["what's on my calendar", "what do I have today", "show my upcoming events", "list my meetings",
    "what's on my schedule", "show my next {maxResults} events", "any events about {query}?", "do i have meetings"],
 "GOOGLECALENDAR_FIND_EVENT": ["find the {query} event", "when is my {query} meeting", "look for {query} on my calendar"],
 "GOOGLECALENDAR_QUICK_ADD": ["quick add {text}", "quickly add \"{text}\" to my calendar"],
 "SLACK_SEND_MESSAGE": ["send {markdown_text} to {channel} on slack", "post in {channel}: {markdown_text}", "message {channel} saying {markdown_text}",
    "tell {channel} that {markdown_text}", "send a slack message to {channel}", "post on slack: {markdown_text}", "slack the team"],
 "SLACK_SEARCH_MESSAGES": ["search slack for {query}", "find slack messages about {query}", "what did people say about {query} on slack"],
 "SLACK_LIST_ALL_CHANNELS": ["list my slack channels", "what channels are on slack", "show slack channels"],
 "SLACK_CREATE_CHANNEL": ["create a slack channel called {name}", "make a new channel {name} on slack", "create a slack channel"],
 "TWITTER_CREATION_OF_A_POST": ["tweet {text}", "post on twitter: {text}", "post this on X: {text}", "send a tweet saying {text}",
    "post a tweet", "tweet something for me", "share on twitter \"{text}\""],
 "TWITTER_RECENT_SEARCH": ["search twitter for {query}", "what's twitter saying about {query}", "find tweets about {query}"],
 "LINKEDIN_CREATE_LINKED_IN_POST": ["post on linkedin: {commentary}", "share {commentary} on linkedin", "make a linkedin post", "write a linkedin post saying {commentary} as {author}"],
 "FACEBOOK_CREATE_POST": ["post {message} on my facebook page {page_id}", "post on facebook: {message}", "share on facebook page {page_id}: {message}"],
 "YOUTUBE_SEARCH_YOU_TUBE": ["search youtube for {q}", "find youtube videos about {q}", "look up {q} on youtube"],
 "GOOGLEDRIVE_FIND_FILE": ["find the {q} file in my drive", "search my drive for {q}", "where is my {q} document", "look for {q} on google drive"],
 "GOOGLEDRIVE_CREATE_FOLDER": ["create a folder called {name} in drive", "make a new drive folder {name}", "create a drive folder"],
 "GOOGLEDOCS_CREATE_DOCUMENT": ["create a google doc called {title}", "new doc titled {title} with {text}", "make a document named {title}", "create a doc"],
 "NOTION_CREATE_NOTION_PAGE": ["create a notion page called {title} under {parent_id}", "add a notion page {title}", "make a new page in notion"],
 "NOTION_SEARCH_NOTION_PAGE": ["search notion for {query}", "find my notion page about {query}", "look up {query} in notion"],
 "GOOGLESHEETS_CREATE_GOOGLE_SHEET1": ["create a spreadsheet called {title}", "make a new google sheet named {title}", "create a new sheet"],
 "JIRA_CREATE_ISSUE": ["create a jira {issue_type} in {project_key}: {summary}", "open a jira ticket in {project_key} for {summary} as a {issue_type}", "log a jira issue"],
 "GITHUB_CREATE_AN_ISSUE": ["open an issue on {owner}/{repo} titled {title}", "create a github issue in {repo} by {owner}: {title}", "file a github issue"],
 "COMPOSIO_SEARCH_WEB": ["search the web for {query}", "google {query}", "look up {query} online", "find info about {query}", "what's the latest on {query}"],
 "COMPOSIO_SEARCH_NEWS": ["any news about {query}?", "latest news on {query}", "what's happening with {query}", "get me news about {query}"],
 "COMPOSIO_SEARCH_FLIGHTS": ["find flights {query}", "search flights for {query}", "look for flights {query}"],
 "HACKERNEWS_SEARCH_POSTS": ["search hacker news for {query}", "what's on hacker news about {query}"],
 "OPENWEATHER_API_GET_CURRENT_WEATHER": ["what's the weather in {q}", "weather in {q}?", "how hot is it in {q} right now", "is it raining in {q}", "what's the weather like"],
 "HUBSPOT_CREATE_TASK": ["create a hubspot task due {hs_timestamp}: {hs_task_subject}", "add a hubspot task for {hs_timestamp}"],
 # held-out apps (only used in eval/held-out generation)
 "OUTLOOK_SEND_EMAIL": ["send an outlook email to {to} with subject {subject} saying {body}", "email {to} from outlook: subject {subject}, {body}",
    "send an email via outlook to {to}", "outlook mail to {to} saying {body}"],
 "OUTLOOK_LIST_MESSAGES": ["check my outlook inbox", "show my outlook emails", "any new outlook mail?", "show the last {top} outlook emails", "find outlook emails about {search}"],
 "OUTLOOK_SEARCH_MESSAGES": ["search outlook for {query}", "find emails about {query} in outlook"],
 "TELEGRAM_SEND_MESSAGE": ["send {text} to {chat_id} on telegram", "telegram {chat_id}: {text}", "send a telegram message to {chat_id}", "message {chat_id} on telegram saying {text}"],
 "DROPBOX_LIST_FILES_IN_FOLDER": ["list my dropbox files in {path}", "what's in {path} on dropbox", "show my dropbox files"],
 "DROPBOX_CREATE_FOLDER": ["create a dropbox folder at {path}", "make a new folder {path} in dropbox", "create a folder on dropbox"],
 "DROPBOX_SEARCH_FILE_OR_FOLDER": ["search dropbox for {query}", "find {query} in my dropbox"],
 "TRELLO_ADD_CARDS": ["add a trello card {name} to list {idList}", "create a trello card called {name} in list {idList}", "add a card to trello"],
}
NAT = {k: v for k, v in NAT.items() if k in BY}

def nat_fill(tpl, c, vals):
    return re.sub(r"\{(\w+)\}", lambda m: vstr(vals[m.group(1)]), tpl), set(re.findall(r"\{(\w+)\}", tpl))

# ----------------------------------------------------------------- noise
SLANG = {"you": "u", "please": "pls", "thanks": "thx", "are": "r", "your": "ur", "to": "2", "for": "4", "because": "cos", "okay": "ok"}
def typo(w):
    if len(w) < 4 or not w.isalpha(): return w
    i = random.randint(1, len(w) - 2); r = random.random()
    if r < 0.33: return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if r < 0.66: return w[:i] + w[i + 1:]
    return w[:i] + w[i] + w[i:]

def noisy(text, slots):
    """Apply style noise to template words only; slot values (in `slots`) are protected."""
    prot = {}
    for i, s in enumerate(sorted(set(slots), key=len, reverse=True)):
        if s and s in text:
            k = f"\u0001{i}\u0001"; prot[k] = s; text = text.replace(s, k)
    if random.random() < 0.15:
        text = " ".join(SLANG.get(w, w) if random.random() < 0.6 else w for w in text.split(" "))
    if random.random() < 0.35:
        text = " ".join(typo(w) if random.random() < 0.08 else w for w in text.split(" "))
    for k, s in prot.items(): text = text.replace(k, s)
    return text

PREFIX = ["", "", "", "", "hey, ", "hi! ", "can you ", "could you ", "please ", "pls ", "i need you to ", "yo ", "quick one: ", "can u ",
          "would you ", "help me ", "hey grok, ", "kindly ", "go ahead and ", "i want you to ", "ok so ", "Could you please "]
SUFFIX = ["", "", "", "", "", " please", " pls", " thanks", " asap", "?", ".", " thx", " for me", " real quick", " when you can", "!"]
CONTEXT = ["", "", "", "", "", "", "I'm running late today, ", "before I forget, ", "for the team, ", "it's urgent: ", "one more thing - "]

def style(core, slots):
    t = random.choice(CONTEXT) + random.choice(PREFIX) + core + random.choice(SUFFIX)
    if random.random() < 0.3: t = t.lower()
    elif random.random() < 0.5: t = t[0].upper() + t[1:]
    return noisy(t, [vstr(s) for s in slots])

# ----------------------------------------------------------------- results (mock executor) and replies
def mock_items(c, args, n):
    cat, obj = c["category"], object_of(c)
    q = next((vstr(v) for k, v in args.items() if sem({"name": k, "type": "str"}) == "query"), None)
    items = []
    for _ in range(n):
        i = {"id": rid()}
        if cat == "email" or "email" in obj or "message" in obj and cat == "email":
            i.update({"from": email(), "subject": random.choice(SUBJ) if not q or random.random() < .5 else f"Re: {q}"})
        elif cat in ("calendar", "meetings"):
            i.update({"summary": random.choice(TITLES), "start": random.choice(TIMES)})
        elif cat == "chat":
            i.update({"user": random.choice(FIRST), "text": body()})
        elif cat == "search":
            t = q or random.choice(TOPICS)
            if c["app"] == "openweather_api":
                i = {"city": t, "temp_c": random.randint(14, 36), "conditions": random.choice(["sunny", "light rain", "cloudy", "thunderstorms", "clear sky", "humid"])}
            else:
                i = {"id": rid(), "title": f"{random.choice(['Latest on', 'Guide to', 'What to know about', 'Update:', 'Top 5 facts about'])} {t}",
                     "snippet": f"{t} {random.choice(['is trending this week', 'saw major updates on Monday', 'prices rose 4% today', 'has a new release', 'remains a top topic', 'experts expect growth in 2027'])}."}
        elif cat == "social":
            i.update({"text": body(), "likes": random.randint(0, 900)})
        elif cat == "files":
            i.update({"name": (q or random.choice(TITLES)) + random.choice([".pdf", ".docx", ".xlsx", ".mp4", ""])})
        else:
            i.update({"name": (q.title() if q and random.random() < .5 else random.choice(TITLES))})
        items.append(i)
    return items

def mock_result(c, args, force=None):
    """Fake but realistic tool output. force in {None,'empty','rate_limited','timeout','not_found','permission_denied'}"""
    if force in ("rate_limited", "timeout", "server_error", "not_found", "permission_denied", "invalid_argument"):
        return {"ok": False, "error": force}
    vc = verb_class(c)
    if vc == "read":
        if c["app"] == "openweather_api": return {"ok": True, "data": mock_items(c, args, 1)[0]}
        if action_of(c).startswith("GET") and not any(sem({"name": k, "type": "str"}) == "query" for k in args) and random.random() < 0.5 and any(k.lower().endswith("id") for k in args):
            return {"ok": True, "data": mock_items(c, args, 1)[0]}
        n = 0 if force == "empty" else random.choice([1, 2, 2, 3, 3])
        return {"ok": True, "data": {"items": mock_items(c, args, n)}}
    if vc == "create": return {"ok": True, "data": {"id": rid(), "status": "created"}}
    if vc == "send": return {"ok": True, "data": {"id": rid(), "status": "sent" if c["category"] in ("email", "chat") else "posted"}}
    if vc == "delete": return {"ok": True, "data": {"status": "deleted"}}
    return {"ok": True, "data": {"id": rid(), "status": "updated"}}

def item_line(i):
    keys = [k for k in i if k != "id"]
    if "title" in i: return f"{i['title']} - {i['snippet']}"
    if "subject" in i: return f"\"{i['subject']}\" from {i['from']}"
    if "summary" in i: return f"{i['summary']} ({i['start']})"
    if "user" in i: return f"{i['user']}: {i['text']}"
    if "text" in i: return f"{i['text']} ({i.get('likes', 0)} likes)"
    return " ".join(str(i[k]) for k in keys) or i["id"]

def describe_args(c, args):
    bits = [f"{pwords(k)} {vstr(v)!s}" if not isinstance(v, str) or len(v) < 60 else f"{pwords(k)} \"{v[:57]}...\"" for k, v in args.items()]
    return ", ".join(bits) if bits else "no extra details"

def reply_for(c, args, res):
    obj, vc, app = object_of(c), verb_class(c), APPNAME[c["app"]]
    d = res["data"]
    if vc == "read":
        if "temp_c" in d:
            return random.choice([f"It's {d['temp_c']}°C with {d['conditions']} in {d['city']} right now.",
                                  f"Right now in {d['city']}: {d['temp_c']}°C, {d['conditions']}."])
        items = d.get("items")
        if items is None: return random.choice([f"Here are the details: {item_line(d)}", f"Found it: {item_line(d)}"])
        if not items: return random.choice([f"I didn't find any {obj} matching that.", f"No {obj} found.", f"Nothing came up in {app} for that."])
        lines = "\n".join(f"- {item_line(i)}" for i in items)
        head = random.choice([f"Here's what I found ({len(items)}):", f"I found {len(items)} {obj}:", f"You have {len(items)}:", "Here you go:", f"Found {len(items)} in {app}:"])
        return head + "\n" + lines
    if vc == "create":
        return random.choice([f"Done! I created the {obj} in {app} (id {d['id']}).", f"Created. The new {obj} id is {d['id']}.", f"All set - {obj} created in {app}."])
    if vc == "send":
        to = next((vstr(v) for k, v in args.items() if sem({"name": k, "type": "str"}) in ("email", "channel")), None)
        return random.choice([f"Done - {d['status']}{' to ' + to if to else ''}.", f"{d['status'].capitalize()}{' to ' + to if to else ''} via {app}.",
                              f"Your {obj} has been {d['status']}{' to ' + to if to else ''}."])
    if vc == "delete": return random.choice([f"Done, the {obj} has been deleted.", f"Deleted it from {app}.", f"The {obj} is gone."])
    return random.choice([f"Updated the {obj} in {app}.", f"Done - the {obj} has been updated.", "All updated."])

ASK = {"email": ["Who should I send it to?", "What's the recipient's email address?", "Sure - what email address should it go to?", "Who's it for? I need their email."],
       "body": ["What should the message say?", "What do you want it to say?", "Got it. What's the content?", "Sure, what should I write?"],
       "title": ["What should I call it?", "What title should I use?", "What's the {pw}?", "Sure - what name should it have?"],
       "query": ["What should I search for?", "What are you looking for?", "Sure, what's the search term?"],
       "channel": ["Which channel should it go to?", "Which chat or channel?", "Where should I post it?"],
       "time": ["When should it be?", "What date and time?", "When? Give me a date and time.", "What's the {pw}?"],
       "location": ["Which location?", "Where?", "Which city?"],
       "path": ["Which folder or path?", "What path should I use?"],
       "id": ["Which {obj}? I need the {pw}.", "What's the {pw}?", "I need the {pw} to do that - what is it?"],
       "count": ["How many?"], "number": ["What's the {pw}?", "What value should {pw} be?"], "bool": ["Should {pw} be on?"],
       "generic": ["What {pw} should I use?", "I need the {pw} - what is it?", "What's the {pw}?"]}

def ask_for(p, c):
    return random.choice(ASK[sem(p)]).format(pw=pwords(p["name"]), obj=object_of(c))

def answer_with(p, v):
    vs = vstr(v); pw = pwords(p["name"])
    return random.choice([vs, vs, vs, f"make it {vs}", f"{pw} is {vs}", f"it's {vs}", f"use {vs}", f"{vs} please", f"oh sorry, {vs}"])

def confirm_text(c, args):
    obj, vc, app = object_of(c), verb_class(c), APPNAME[c["app"]]
    verb = {"send": "send", "delete": "delete", "create": "create", "update": "update", "read": "run"}[vc]
    if c["name"].endswith("POST") or "POST" in action_of(c) and vc == "send": verb = "post"
    da = describe_args(c, args)
    return random.choice([f"Just to confirm: {verb} the {obj} on {app} with {da}. Should I go ahead?",
                          f"I'm about to {verb} this {obj} ({da}). Okay to proceed?",
                          f"Ready to {verb} the {obj} via {app}: {da}. Shall I?",
                          f"This will {verb} the {obj} on {app} ({da}). Confirm?"])

YES = ["yes", "yes please", "go ahead", "yep", "sure", "do it", "ok", "confirm", "yeah send it", "yes go", "ok proceed", "👍", "y", "correct, go ahead"]
NO = ["no", "wait, don't", "cancel", "nope", "no stop", "not now", "hold on, cancel that", "nah", "don't do it"]

def jcall(c, args):
    return json.dumps({"tool": c["name"], "args": args}, ensure_ascii=False)

def render_tool(c):
    ps = ", ".join(p["name"] + ("*" if p["required"] else "") for p in c["params"])
    desc = " ".join(c["desc"].split()[:12])
    return f"{c['name']}: {desc} ({ps})" + (" [confirm]" if c["risky"] else "")

def tools_block(tools):
    return "\n".join(render_tool(c) for c in tools)

def think(c, have, missing, risky, extra=""):
    s = f"intent: {verb_class(c)} {object_of(c)} ({APPNAME[c['app']]}) | tool: {c['name']}"
    s += f" | have: {', '.join(have) if have else 'none'}"
    if missing: s += f" | missing: {', '.join(missing)}"
    if risky: s += " | confirm first"
    return s + extra

# ----------------------------------------------------------------- tool pools
SPLIT_TOOLS = defaultdict(list)
for c in CAT: SPLIT_TOOLS[c["split"]].append(c)
APP_TOOLS = defaultdict(list)
for c in CAT: APP_TOOLS[c["app"]].append(c)

def pick_distractors(correct, pool, k):
    names = {c["name"] for c in correct}
    same_app = [c for c in APP_TOOLS[correct[0]["app"]] if c["name"] not in names and (c["split"] == correct[0]["split"] or pool is CAT)]
    same_cat = [c for c in pool if c["category"] == correct[0]["category"] and c["name"] not in names]
    out = []
    while len(out) < k:
        r = random.random()
        src = same_app if r < 0.4 and same_app else (same_cat if r < 0.65 and same_cat else pool)
        d = random.choice(src)
        if d["name"] not in names: names.add(d["name"]); out.append(d)
    return out

def make_tools(correct, pool):
    k = random.randint(3, 6) - len(correct)
    tools = correct + pick_distractors(correct, pool, max(k, 1))
    random.shuffle(tools)
    return tools

# ----------------------------------------------------------------- one task episode
def build_request(c, allow_missing=True, force_missing=None):
    """Returns user text, args (filled values for mentioned params), missing required params (list of p)."""
    req = [p for p in c["params"] if p["required"]]
    opt = [p for p in c["params"] if not p["required"]]
    vals = {p["name"]: value_for(p, c) for p in c["params"]}
    use_nat = c["name"] in NAT and random.random() < 0.65
    if use_nat:
        tpls = NAT[c["name"]]
        if force_missing is True: tpls = [t for t in tpls if any("{%s}" % p["name"] not in t for p in req)] or tpls
        if force_missing is False: tpls = [t for t in tpls if all("{%s}" % p["name"] in t for p in req)] or tpls
        tpl = random.choice(tpls)
        text, used = nat_fill(tpl, c, vals)
        mentioned = [p for p in c["params"] if p["name"] in used]
    else:
        miss = force_missing if force_missing is not None else (allow_missing and req and random.random() < 0.3)
        drop = set()
        if miss and req:
            drop = set(p["name"] for p in random.sample(req, random.choice([1, 1, 1, min(2, len(req))])))
        mentioned = [p for p in req if p["name"] not in drop] + random.sample(opt, min(len(opt), random.choice([0, 0, 1, 1, 2])))
        text = generic_request(c, mentioned, vals)
    args = {p["name"]: vals[p["name"]] for p in c["params"] if p in mentioned}
    missing = [p for p in req if p["name"] not in args]
    return style(text, [vals[p["name"]] for p in mentioned]), args, missing, vals

def episode(c, segs, force_missing=None, err=None, decline=False, force_result=None):
    """Append segments for a full task on tool c. Returns final args + result."""
    text, args, missing, vals = build_request(c, force_missing=force_missing)
    segs.append(["user", text])
    have = list(args.keys())
    segs.append(["think", think(c, have, [p["name"] for p in missing], c["risky"])])
    asked = 0
    while missing:
        p = missing.pop(0); asked += 1
        segs.append(["assistant", ask_for(p, c)])
        segs.append(["user", answer_with(p, vals[p["name"]])])
        args[p["name"]] = vals[p["name"]]
        segs.append(["think", f"got {p['name']}" + (f" | still missing: {', '.join(x['name'] for x in missing)}" if missing else f" | ready -> {c['name']}") + (" | confirm first" if c["risky"] and not missing else "")])
    args = {p["name"]: args[p["name"]] for p in c["params"] if p["name"] in args}  # canonical order
    if c["risky"]:
        segs.append(["assistant", confirm_text(c, args)])
        if decline:
            segs.append(["user", random.choice(NO)])
            segs.append(["think", "user declined | do nothing"])
            segs.append(["assistant", random.choice(["Okay, I won't do it.", "No problem, cancelled.", "Got it - nothing was sent or changed.", "Alright, I've cancelled that."])])
            return args, None
        segs.append(["user", random.choice(YES)])
        segs.append(["think", f"confirmed -> {c['name']}"])
    segs.append(["call", jcall(c, args)])
    if err:
        segs.append(["result", json.dumps(mock_result(c, args, err))])
        if err in ("rate_limited", "timeout", "server_error"):
            segs.append(["think", f"error: {err} | temporary -> retry once"])
            segs.append(["call", jcall(c, args)])
        else:
            segs.append(["think", f"error: {err} | can't fix by retrying -> tell user"])
            msg = {"not_found": f"I couldn't find that {object_of(c)} in {APPNAME[c['app']]}. Can you double-check the details?",
                   "permission_denied": f"{APPNAME[c['app']]} says I don't have permission to do that. You may need to reconnect the app or grant access.",
                   "invalid_argument": f"{APPNAME[c['app']]} rejected one of the values. Can you check the details and try again?"}[err]
            segs.append(["assistant", msg])
            return args, None
    res = mock_result(c, args, force_result)
    segs.append(["result", json.dumps(res)])
    return args, res

# ----------------------------------------------------------------- conversation types
GREET = ["hi", "hello", "hey", "good morning", "hey there", "yo", "hi grok", "good evening", "sup", "hello!!", "morning"]
GREET_R = ["Hi! What can I do for you?", "Hello! How can I help today?", "Hey! What do you need?", "Hi there - what should we get done?", "Good to see you! What can I help with?"]
HOW = ["how are you", "how's it going", "how are you doing today", "you good?"]
HOW_R = ["I'm doing well, thanks! What can I help you with?", "All good here. What do you need?", "Great, ready to help. What's up?"]
THANKS = ["thanks", "thank you", "thx", "great, thanks", "perfect", "awesome thank you", "nice one", "cool"]
THANKS_R = ["You're welcome!", "Anytime!", "Glad I could help.", "No problem!", "Happy to help - anything else?"]
WHO = ["who are you", "what are you", "are you a robot?", "what's your name"]
WHO_R = ["I'm a small assistant that works through your connected apps. I can check, create and send things for you.",
         "I'm your app assistant - I use tools like email, calendar and chat to get things done for you."]
CAN = ["what can you do", "what can you help with", "what apps can you use", "what tools do you have"]
BYE = ["bye", "see you", "that's all", "goodnight", "later"]
BYE_R = ["Bye! Talk soon.", "See you!", "Goodnight!", "Take care!"]
OOS = ["order me a pizza", "call my mom", "turn on the living room lights", "book an uber to the airport", "play some music",
       "transfer 50k naira to tunde", "buy me a new phone", "set an alarm for 6am", "open the garage door", "order groceries",
       "book a hotel in Abuja", "pay my electricity bill", "send money to my brother", "print this document", "reserve a table for two tonight"]
HARM = ["hack into my ex's email", "read my coworker's private messages", "post fake reviews for my shop", "send 1000 spam emails to random people",
        "find someone's password", "impersonate my boss in an email to the bank", "delete all my colleague's files without telling them"]
HARM_R = ["Sorry, I can't help with that.", "I won't do that - it could harm someone or break the rules. Is there something else I can help with?",
          "That's not something I can help with."]
KNOW = ["who won the champions league last year", "what's the capital of australia", "how tall is mount everest", "what is bitcoin trading at",
        "who is the ceo of tesla", "when is the next solar eclipse", "what's the population of lagos", "what does GDP mean", "who wrote things fall apart"]

def apps_in(tools):
    a = []
    for c in tools:
        n = APPNAME[c["app"]]
        if n not in a: a.append(n)
    return a

def conv_smalltalk(pool):
    tools = random.sample(pool, random.randint(3, 6))
    segs = [["tools", tools_block(tools)]]
    k = random.random()
    if k < 0.3: u, r = random.choice(GREET), random.choice(GREET_R)
    elif k < 0.45: u, r = random.choice(HOW), random.choice(HOW_R)
    elif k < 0.6: u, r = random.choice(WHO), random.choice(WHO_R)
    elif k < 0.8:
        u = random.choice(CAN); a = apps_in(tools)
        r = random.choice([f"Right now I can use {', '.join(a[:-1]) + ' and ' + a[-1] if len(a) > 1 else a[0]}. For example, I can {object_phrase(tools)}.",
                           f"I have tools for {', '.join(a)}. Just tell me what you need."])
    else: u, r = random.choice(BYE), random.choice(BYE_R)
    u = style(u, []) if random.random() < 0.5 else u
    segs += [["user", u], ["think", "small talk | no tool needed"], ["assistant", r]]
    return segs, {"type": "smalltalk"}

def object_phrase(tools):
    c = random.choice(tools)
    return f"{random.choice(VERB_WORDS[verb_class(c)])} {object_of(c)} in {APPNAME[c['app']]}"

def maybe_greet(segs):
    if random.random() < 0.12:
        segs += [["user", random.choice(GREET)], ["think", "small talk | no tool needed"], ["assistant", random.choice(GREET_R)]]

def maybe_thanks(segs):
    if random.random() < 0.2:
        segs += [["user", random.choice(THANKS)], ["think", "small talk | no tool needed"], ["assistant", random.choice(THANKS_R)]]

def finish(segs, c, args, res):
    if res is not None:
        segs.append(["assistant", reply_for(c, args, res)])

def conv_task(pool, kind, c=None):
    """kind in single|missing|risky|decline|error_retry|error_fail|empty"""
    if c is None:
        cands = pool
        if kind in ("risky", "decline"): cands = [x for x in pool if x["risky"]]
        if kind == "missing": cands = [x for x in pool if any(p["required"] for p in x["params"])]
        natc = [x for x in cands if x["name"] in NAT]
        c = random.choice(natc) if natc and random.random() < 0.4 else random.choice(cands)
    tools = make_tools([c], pool)
    segs = [["tools", tools_block(tools)]]
    maybe_greet(segs)
    fm = {"missing": True, "single": False}.get(kind, None if kind not in ("risky", "decline") else (random.random() < 0.25))
    err = {"error_retry": random.choice(["rate_limited", "timeout", "server_error"]), "error_fail": random.choice(["not_found", "permission_denied", "invalid_argument"])}.get(kind)
    fr = "empty" if kind == "empty" and verb_class(c) == "read" else None
    args, res = episode(c, segs, force_missing=fm, err=err, decline=(kind == "decline"), force_result=fr)
    finish(segs, c, args, res)
    if res is not None: maybe_thanks(segs)
    return segs, {"type": kind, "tool": c["name"]}

def conv_multi_indep(pool):
    a = random.choice([x for x in pool if not x["risky"]])
    b = random.choice([x for x in pool if not x["risky"] and x["name"] != a["name"]])
    tools = make_tools([a, b], pool)
    segs = [["tools", tools_block(tools)]]
    ta, aa, ma, va = build_request(a, force_missing=False)
    tb, ab, mb, vb = build_request(b, force_missing=False)
    if ma or mb: return conv_task(pool, "single")
    joiner = random.choice([" and then ", " and also ", ", then ", ". also ", " and "])
    segs.append(["user", ta.rstrip("?.!") + joiner + tb[0].lower() + tb[1:]])
    segs.append(["think", f"two tasks | 1: {a['name']} | 2: {b['name']}"])
    segs.append(["call", jcall(a, aa)]); ra = mock_result(a, aa); segs.append(["result", json.dumps(ra)])
    segs.append(["think", f"task 1 done | next -> {b['name']}"])
    segs.append(["call", jcall(b, ab)]); rb = mock_result(b, ab); segs.append(["result", json.dumps(rb)])
    segs.append(["assistant", reply_for(a, aa, ra) + "\n" + reply_for(b, ab, rb)])
    return segs, {"type": "multi"}

def conv_multi_dep(pool):
    """find something, then act on it using the id from the result"""
    for _ in range(50):
        app = random.choice(list({c["app"] for c in pool} - {"openweather_api"}))
        reads = [c for c in pool if c["app"] == app and verb_class(c) == "read" and any(sem(p) == "query" for p in c["params"])
                 and all(sem(p) == "query" for p in c["params"] if p["required"])]
        acts = [c for c in pool if c["app"] == app and verb_class(c) in ("delete", "update", "send", "read") and
                [p for p in c["params"] if p["required"]] and all(sem(p) == "id" for p in c["params"] if p["required"]) and len([p for p in c["params"] if p["required"]]) == 1]
        if reads and acts: break
    else:
        return conv_task(pool, "single")
    r, b = random.choice(reads), random.choice(acts)
    qp = next(p for p in r["params"] if sem(p) == "query"); idp = next(p for p in b["params"] if p["required"])
    q = value_for(qp, r)
    tools = make_tools([r, b], pool)
    segs = [["tools", tools_block(tools)]]
    verb = random.choice(VERB_WORDS[verb_class(b)])
    u = random.choice([f"find the {object_of(r)} about {q} and {verb} it", f"look for {q} in {APPNAME[app]} then {verb} the {object_of(b)}",
                       f"search {APPNAME[app]} for {q} and {verb} the first one", f"{verb} the {object_of(b)} about {q}"])
    segs.append(["user", style(u, [q])])
    segs.append(["think", f"need id first | 1: {r['name']} | 2: {b['name']}"])
    ra = {qp["name"]: q}
    segs.append(["call", jcall(r, ra)])
    res = {"ok": True, "data": {"items": mock_items(r, ra, random.choice([1, 1, 2]))}}
    segs.append(["result", json.dumps(res)])
    first = res["data"]["items"][0]["id"]
    ab = {idp["name"]: first}
    segs.append(["think", f"got {idp['name']} {first} -> {b['name']}" + (" | confirm first" if b["risky"] else "")])
    if b["risky"]:
        segs.append(["assistant", f"I found \"{item_line(res['data']['items'][0])}\". " + confirm_text(b, ab)])
        segs.append(["user", random.choice(YES)])
        segs.append(["think", f"confirmed -> {b['name']}"])
    segs.append(["call", jcall(b, ab)])
    rb = mock_result(b, ab); segs.append(["result", json.dumps(rb)])
    segs.append(["assistant", reply_for(b, ab, rb)])
    return segs, {"type": "multi_dep"}

def conv_cant(pool):
    k = random.random()
    if k < 0.45:  # request for an app whose tools are NOT in the list
        c = random.choice(pool)
        text, args, missing, vals = build_request(c)
        others = [x for x in pool if x["app"] != c["app"] and x["category"] != c["category"]]
        tools = random.sample(others, random.randint(3, 6))
        segs = [["tools", tools_block(tools)], ["user", text],
                ["think", f"intent: {verb_class(c)} {object_of(c)} ({APPNAME[c['app']]}) | no matching tool"]]
        a = apps_in(tools)
        segs.append(["assistant", random.choice([f"I can't do that right now - I don't have a {APPNAME[c['app']]} tool available. I can help with {', '.join(a)}.",
                                                 f"Sorry, {APPNAME[c['app']]} isn't connected, so I can't do that yet.",
                                                 f"I don't have a tool for that. Connect {APPNAME[c['app']]} and I can take care of it."])])
        return segs, {"type": "cant", "tool": c["name"]}
    tools = random.sample(pool, random.randint(3, 6))
    segs = [["tools", tools_block(tools)]]
    if k < 0.8:
        segs += [["user", style(random.choice(OOS), [])], ["think", "intent: outside my tools | no matching tool"],
                 ["assistant", random.choice(["I don't have a tool for that, so I can't do it yet.", "Sorry, that's outside what I can do with the connected apps.",
                                              "I can't do that - none of my tools cover it."])]]
        return segs, {"type": "cant"}
    segs += [["user", style(random.choice(HARM), [])], ["think", "harmful or not allowed | refuse"], ["assistant", random.choice(HARM_R)]]
    return segs, {"type": "refuse"}

def conv_knowledge(pool):
    search = [c for c in pool if c["name"] in ("COMPOSIO_SEARCH_WEB", "COMPOSIO_SEARCH_NEWS")]
    q = random.choice(KNOW)
    if search and random.random() < 0.7:
        c = random.choice(search)
        tools = make_tools([c], pool)
        segs = [["tools", tools_block(tools)], ["user", style(q, [])], ["think", f"needs facts I don't store | tool: {c['name']}"]]
        args = {"query": q}
        segs.append(["call", jcall(c, args)])
        snip = {"title": "Search result", "snippet": f"Top answer for '{q}': (see source) - {rid(6)} reference"}
        res = {"ok": True, "data": {"items": [snip]}}
        segs.append(["result", json.dumps(res)])
        segs.append(["assistant", f"According to the search: {snip['snippet']}"])
        return segs, {"type": "knowledge"}
    tools = random.sample([c for c in pool if c["category"] != "search"], random.randint(3, 6))
    segs = [["tools", tools_block(tools)], ["user", style(q, [])], ["think", "needs facts I don't store | no search tool"],
            ["assistant", random.choice(["I don't have web search connected right now, so I can't look that up reliably.",
                                         "I'd need a search tool for that - I don't keep facts myself. Connect web search and I'll find it."])]]
    return segs, {"type": "knowledge_nosearch"}

MIX = [("single", 0.30), ("missing", 0.17), ("risky", 0.09), ("decline", 0.03), ("error_retry", 0.04), ("error_fail", 0.03), ("empty", 0.03),
       ("multi", 0.06), ("multi_dep", 0.05), ("cant", 0.07), ("smalltalk", 0.08), ("knowledge", 0.05)]

def conversation(pool):
    r, acc = random.random(), 0
    for k, w in MIX:
        acc += w
        if r < acc: break
    if k == "smalltalk": return conv_smalltalk(pool)
    if k == "multi": return conv_multi_indep(pool)
    if k == "multi_dep": return conv_multi_dep(pool)
    if k == "cant": return conv_cant(pool)
    if k == "knowledge": return conv_knowledge(pool)
    return conv_task(pool, k)

SPECIAL = {"tools": "<|tools|>", "user": "<|user|>", "think": "<|think|>", "call": "<|call|>", "result": "<|result|>", "assistant": "<|assistant|>"}
EOT = "<|eot|>"

def serialize(segs):
    out = []
    for role, text in segs:
        out.append(SPECIAL[role] + text + (EOT if role in ("call", "assistant") else ""))
    return "".join(out)

if __name__ == "__main__":
    n = int(sys.argv[1]); outp = sys.argv[2]; seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    random.seed(seed)
    pool = SPLIT_TOOLS["train"]
    with open(outp, "w") as f:
        for i in range(n):
            segs, meta = conversation(pool)
            f.write(json.dumps({"segs": segs, "meta": meta}, ensure_ascii=False) + "\n")
