"""
RAG Part 1 - Visualizer
=======================

This app shows, step by step, what "Part 1" of a RAG pipeline does
with the essays in the `new` folder.

Part 1 has 4 steps:
  1. READ    - open each .docx file and take out the plain text
  2. CUT     - cut each essay into small pieces ("chunks")
  3. NUMBERS - turn each piece into a list of numbers that stands for its
               meaning (an "embedding")
  4. STORE   - save the pieces + numbers so they can be searched fast

Then there is a CHAT (tab 4): you talk with Claude (an AI at Anthropic) in
sessions. Claude remembers the whole session, and with every message it gets:
  - the pieces of your essays closest to the message (found as in Part 2),
  - your WORLDVIEW GUIDE (worldview.md, tab 5) - a summary of your beliefs,
    key words and style that Claude made by reading all your essays,
  - your writing rules (writing_rules.md).
Sessions are stored as pieces + meaning numbers, like the essays. After a
session, Claude can suggest points for your worldview guide - only the ones
you approve are added. The chat needs an API key and sends data over the
internet.

How to start it:  double-click run.bat  (or see README.md)
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from docx import Document
from sklearn.manifold import TSNE

# ---------------------------------------------------------------------------
# Settings (you can change these)
# ---------------------------------------------------------------------------

# Where the essays are. This app only READS this folder. It never changes it.
DOCS_FOLDER = Path(__file__).resolve().parent.parent / "new"

# Where we save our work, so we don't redo it every time.
# (This is our simple "vector database".)
STORE_FOLDER = Path(__file__).resolve().parent / "store"

# About how many words go into one piece.
WORDS_PER_PIECE = 150

# The free model that turns text into "meaning numbers".
# It runs on this laptop. Nothing is sent to the internet
# (only the model itself is downloaded once, the first time).
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The AI that WRITES the answer (step 5). This one is NOT on your laptop:
# it runs at Anthropic, so it needs an API key and an internet connection.
CLAUDE_MODEL = "claude-opus-5-5"   # the one picked at the start

# The Claude models you can pick from, with their price in US dollars per
# 1 million tokens (input = what Claude reads, output = what it writes,
# including its thinking). Prices from Anthropic, November 2026.
MODELS = {
    "claude-fable-5-1":  {"name": "Fable 5.1 - the most capable, most expensive", "in": 10.0, "out": 50.0},
    "claude-opus-5-5":   {"name": "Opus 5.5 - very capable (recommended)",        "in": 4.0,  "out": 20.0},
    "claude-sonnet-5-5": {"name": "Sonnet 5.5 - good and half the price of Opus", "in": 2.0,  "out": 10.0},
    "claude-haiku-5-5":  {"name": "Haiku 5.5 - fastest and cheapest",             "in": 0.10, "out": 0.50,
                          # Haiku costs more when it reads over 100,000 tokens at once.
                          "big_in": 0.50, "big_out": 2.50},
}

# How hard Claude thinks before it writes. More effort = better on hard
# topics, but slower and costs more (thinking counts as output).
EFFORTS = {
    "low":    "Low - quick, little thinking",
    "medium": "Medium - a good balance",
    "high":   "High - thinks carefully",
    "xhigh":  "Extra high - thinks a lot",
    "max":    "Max - thinks as much as it needs (slowest, most expensive)",
}

# Where you can put your Claude API key (one line, nothing else).
# The app also accepts the ANTHROPIC_API_KEY environment variable.
KEY_FILE = Path(__file__).resolve().parent / "claude_key.txt"

# The worldview guide (made in tab 5). It is a normal text file: you can open
# it in Notepad, read it and correct it. It is kept OUTSIDE the `store`
# folder on purpose, so deleting `store` never deletes your guide.
WORLDVIEW_FILE = Path(__file__).resolve().parent / "worldview.md"
WORLDVIEW_OLD_FILE = Path(__file__).resolve().parent / "worldview_old.md"  # the guide before the last rebuild
WORLDVIEW_INFO_FILE = Path(__file__).resolve().parent / "worldview_info.json"  # when it was made, from what

# Your standing writing rules (Claude follows them every time). Edit in Notepad.
WRITING_RULES_FILE = Path(__file__).resolve().parent / "writing_rules.md"

# The chat (tab 4). Sessions are saved in the `store` folder as pieces +
# meaning numbers, like the essays.
CHAT_SESSIONS_FILE = STORE_FOLDER / "chat_sessions.pkl"   # one row per session
CHAT_MESSAGES_FILE = STORE_FOLDER / "chat_messages.pkl"   # every message, in order
CHAT_PIECES_FILE = STORE_FOLDER / "chat_pieces.pkl"       # YOUR words, cut into pieces
CHAT_NUMBERS_FILE = STORE_FOLDER / "chat_numbers.npy"     # their meaning numbers
CHAT_INFO_FILE = STORE_FOLDER / "chat_info.json"          # which meaning model made them

# How many essay pieces Claude gets with each message.
CHAT_TOP_K = 8
# Up to this many things you said in OTHER sessions (or approved worldview
# points) are added too, if they are at least this close to the message.
PAST_TOP_K = 3
PAST_MIN_SCORE = 0.7

# The old History folder. Its answers and essays are moved into chat sessions
# once, then it is renamed history_moved_backup.
OLD_HISTORY_FOLDER = Path(__file__).resolve().parent / "history"


# ---------------------------------------------------------------------------
# STEP 1 - READ: take the plain text out of a .docx file
# ---------------------------------------------------------------------------

def read_docx(path: Path) -> list[str]:
    """Open a Word file and give back its paragraphs as plain text.
    Pictures, fonts and colours are thrown away. Empty lines are skipped."""
    try:
        doc = Document(str(path))
    except Exception:
        # The file is empty or broken, so there is nothing to read.
        return []
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


# ---------------------------------------------------------------------------
# STEP 2 - CUT: cut one essay into small pieces
# ---------------------------------------------------------------------------

def cut_into_pieces(paragraphs: list[str], words_per_piece: int) -> list[str]:
    """Join paragraphs together until a piece has about `words_per_piece`
    words, then start a new piece. We never cut in the middle of a
    paragraph, so each piece stays easy to read."""
    pieces, current, count = [], [], 0
    for para in paragraphs:
        current.append(para)
        count += len(para.split())
        if count >= words_per_piece:
            pieces.append("\n\n".join(current))
            current, count = [], 0
    if current:  # whatever is left over becomes the last piece
        pieces.append("\n\n".join(current))
    return pieces


# ---------------------------------------------------------------------------
# STEP 3 - NUMBERS: the model that turns text into "meaning numbers"
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the meaning model (first time downloads ~100 MB)...")
def load_model():
    from fastembed import TextEmbedding
    return TextEmbedding(MODEL_NAME)


def to_numbers(model, texts: list[str]) -> np.ndarray:
    """Turn each text into a list of 384 numbers.
    We also 'normalise' them (make each list the same length), so that
    comparing two lists gives a simple closeness score from about 0 to 1."""
    vecs = np.array(list(model.embed(texts)), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


# ---------------------------------------------------------------------------
# STEP 4 - STORE: save everything, and load it back next time
# ---------------------------------------------------------------------------

def folder_fingerprint(files: list[Path]) -> str:
    """A short code that changes whenever any essay is added, removed or
    edited. If the code is the same as last time, we can reuse our saved work."""
    h = hashlib.md5()
    for f in files:
        s = f.stat()
        h.update(f"{f.name}|{s.st_size}|{s.st_mtime}|{WORDS_PER_PIECE}|{MODEL_NAME}".encode())
    return h.hexdigest()


def file_key(f: Path) -> str:
    """A short 'ID card' for one file: its size + when it was last saved.
    If either changes, we know the file was edited."""
    s = f.stat()
    return f"{s.st_size}|{s.st_mtime_ns}"


def load_saved_library():
    """Load what we saved last time. Gives back empty tables if there is
    nothing saved, or if the settings (model / piece size) have changed,
    because then the old numbers can't be mixed with new ones."""
    meta_file = STORE_FOLDER / "info.json"
    docs_file = STORE_FOLDER / "documents.csv"
    pieces_file = STORE_FOLDER / "pieces.csv"
    vecs_file = STORE_FOLDER / "numbers.npy"
    empty = (pd.DataFrame(), pd.DataFrame(), np.zeros((0, 0), dtype=np.float32))
    if not (meta_file.exists() and docs_file.exists() and pieces_file.exists() and vecs_file.exists()):
        return empty
    meta = json.loads(meta_file.read_text())
    if meta.get("model") != MODEL_NAME or meta.get("words_per_piece") != WORDS_PER_PIECE:
        return empty
    docs = pd.read_csv(docs_file)
    if "file key" not in docs.columns:  # saved by an older version of this app
        return empty
    pieces = pd.read_csv(pieces_file)
    vecs = np.load(vecs_file)
    if len(pieces) != len(vecs):  # saved files don't match - safer to start fresh
        return empty
    return docs, pieces, vecs


@st.cache_data(show_spinner="Updating the library (only new or changed essays)...")
def build_library(fingerprint: str):
    """Runs the 4 steps of Part 1, but ONLY for essays that are new or changed.
    Essays we already did are reused from the `store` folder.

    Gives back:
      docs    - one row per essay (name, words, pieces, used or skipped)
      pieces  - one row per piece (essay name, piece number, text, x, y)
      vecs    - the meaning numbers for every piece
      report  - what was done this time (reused / new / changed / removed)
    """
    STORE_FOLDER.mkdir(exist_ok=True)
    old_docs, old_pieces, old_vecs = load_saved_library()

    # SHORTCUT: if no essay at all changed since last time, just use the saved
    # library as it is (including the map), so the app opens instantly.
    meta_file = STORE_FOLDER / "info.json"
    if len(old_docs) and json.loads(meta_file.read_text()).get("fingerprint") == fingerprint:
        report = {"reused": old_docs["essay"].tolist(), "new": [], "changed": [], "removed": []}
        return old_docs, old_pieces, old_vecs, report

    # What did we save last time?  essay name -> its file ID card
    old_keys = dict(zip(old_docs["essay"], old_docs["file key"])) if len(old_docs) else {}

    files = sorted(f for f in DOCS_FOLDER.glob("*.docx") if not f.name.startswith("~$"))
    current_names = {f.stem for f in files}

    doc_rows, piece_rows, vec_parts = [], [], []
    report = {"reused": [], "new": [], "changed": [], "removed": []}
    model = None  # only load the model if we actually need it

    for f in files:
        name, key = f.stem, file_key(f)

        if old_keys.get(name) == key:
            # SAME AS LAST TIME -> reuse the saved pieces and numbers. No work needed.
            report["reused"].append(name)
            doc_rows.append(old_docs[old_docs["essay"] == name].iloc[0].to_dict())
            mask = (old_pieces["essay"] == name).values
            for row in old_pieces[mask].to_dict("records"):
                piece_rows.append({k: row[k] for k in ("essay", "piece", "words", "text")})
            vec_parts.append(old_vecs[mask])
            continue

        # NEW or CHANGED -> do the real work for this essay only.
        report["changed" if name in old_keys else "new"].append(name)

        # STEP 1 - read, STEP 2 - cut
        paragraphs = read_docx(f)
        words = sum(len(p.split()) for p in paragraphs)
        texts = cut_into_pieces(paragraphs, WORDS_PER_PIECE) if words else []
        doc_rows.append({
            "essay": name,
            "file size (KB)": round(f.stat().st_size / 1024),
            "words": words,
            "pieces": len(texts),
            "status": "used" if texts else "skipped (no text)",
            "file key": key,
        })
        for i, text in enumerate(texts, start=1):
            piece_rows.append({"essay": name, "piece": i,
                               "words": len(text.split()), "text": text})

        # STEP 3 - numbers (only for this essay's pieces)
        if texts:
            model = model or load_model()
            vec_parts.append(to_numbers(model, texts))

    # Essays that were saved before but are no longer in the folder.
    report["removed"] = sorted(set(old_keys) - current_names)

    docs = pd.DataFrame(doc_rows)
    pieces = pd.DataFrame(piece_rows)
    vecs = np.vstack(vec_parts) if vec_parts else np.zeros((0, 384), dtype=np.float32)

    # For the MAP only: squeeze 384 numbers down to 2 (x and y),
    # trying to keep pieces with similar meaning close together.
    # This is redone every time something changes. It is quick (a few seconds).
    # (This is just for drawing. Searching always uses all 384 numbers.)
    tsne = TSNE(n_components=2, metric="cosine", init="pca",
                perplexity=min(30, len(pieces) - 1), random_state=42)
    xy = tsne.fit_transform(vecs)
    pieces["x"], pieces["y"] = xy[:, 0], xy[:, 1]

    # STEP 4 - store: save it all for next time.
    docs.to_csv(STORE_FOLDER / "documents.csv", index=False)
    pieces.to_csv(STORE_FOLDER / "pieces.csv", index=False)
    np.save(STORE_FOLDER / "numbers.npy", vecs)
    (STORE_FOLDER / "info.json").write_text(json.dumps(
        {"fingerprint": fingerprint, "model": MODEL_NAME, "words_per_piece": WORDS_PER_PIECE}))
    return docs, pieces, vecs, report


# ---------------------------------------------------------------------------
# STEP 5 - ANSWER: Claude writes the answer from the best pieces
# (This is the last part of "Part 2". Steps 1-4 above are "Part 1".)
# ---------------------------------------------------------------------------

def get_claude_key() -> str | None:
    """Find the API key: first the environment variable, then claude_key.txt."""
    import os
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text(encoding="utf-8-sig").strip()
    return key or None


def has_claude_key() -> bool:
    return get_claude_key() is not None


def number_the_pieces(chosen: pd.DataFrame) -> str:
    """Put the pieces in a numbered list, so Claude can point to them as [1], [2], ..."""
    return "\n\n".join(
        f"[{n}] (from the essay \"{r.essay}\", piece {r.piece})\n{r.text}"
        for n, r in enumerate(chosen.itertuples(), start=1))


def cost_in_dollars(model: str, tokens_in: int, tokens_out: int,
                    cache_write: int = 0, cache_read: int = 0) -> float:
    """What one request cost, from the price list in MODELS.
    Reading from the cache costs 1/10 of the normal price; putting something
    into the cache costs 1.25 times the normal price."""
    p = MODELS.get(model, MODELS[CLAUDE_MODEL])
    price_in, price_out = p["in"], p["out"]
    if tokens_in + cache_write + cache_read > 100_000 and "big_in" in p:
        price_in, price_out = p["big_in"], p["big_out"]
    read = tokens_in + 1.25 * cache_write + 0.1 * cache_read
    return (read * price_in + tokens_out * price_out) / 1e6


def room_for(effort: str) -> int:
    """Room for thinking + writing. Higher effort thinks more, so it gets more
    room. (You only pay for what is really used.)"""
    return {"low": 16000, "medium": 16000, "high": 32000}.get(effort, 64000)


def usage_note(msg, effort: str, model: str) -> str:
    """A short note: which model wrote it, how many tokens, what it cost."""
    u = msg.usage
    cw = u.cache_creation_input_tokens or 0
    cr = u.cache_read_input_tokens or 0
    tin, tout = u.input_tokens + cw + cr, u.output_tokens
    cost = cost_in_dollars(msg.model, u.input_tokens, tout, cw, cr)
    note = (f"Written by {msg.model}, effort {effort}. It read {tin:,} tokens"
            f"{f' ({cr:,} of them from the cache, which is cheaper)' if cr else ''} and "
            f"wrote {tout:,} tokens, including its thinking (a token is about 3/4 of "
            f"a word). Cost: about {cost * 100:.1f} US cents.")
    if msg.model != model:
        note += f" (You picked {model}, but Anthropic used {msg.model} as a safety fallback.)"
    return note


def choose_model(key: str, default_effort: str) -> tuple[str, str]:
    """Show the 'Model' and 'Effort' pickers. Gives back (model, effort)."""
    # The starting choice is put in Session State (not given as `index`), so the
    # app can also set it later, e.g. when you open a chat session.
    st.session_state.setdefault(f"{key}_model", CLAUDE_MODEL)
    st.session_state.setdefault(f"{key}_effort", default_effort)
    c1, c2 = st.columns(2)
    model = c1.selectbox("Model", list(MODELS),
                         format_func=lambda m: MODELS[m]["name"], key=f"{key}_model",
                         help="Price per 1 million tokens (read / written): " + ", ".join(
                             f"{m.replace('claude-', '')} ${p['in']:g} / ${p['out']:g}"
                             for m, p in MODELS.items()))
    effort = c2.selectbox("Effort", list(EFFORTS),
                          format_func=EFFORTS.get, key=f"{key}_effort",
                          help="How hard Claude thinks before writing. Higher effort is "
                               "slower and costs more, because thinking counts as written tokens.")
    return model, effort


def ask_claude(instructions: str, prompt: str, effort: str,
               model: str = CLAUDE_MODEL,
               messages: list[dict] | None = None) -> tuple[str, str] | None:
    """Send a request to Claude and show the reply on the screen word by word
    as it is written ("streaming"). Used for answers, essays and the worldview guide.
    For the chat, `messages` is the whole conversation so far (then the
    conversation is also "cached" at Anthropic, so the next message, which
    starts with the same conversation, is cheaper to read).
    Gives back (the full text, a short note about it), or None if it failed."""
    import anthropic

    client = anthropic.Anthropic(api_key=get_claude_key())
    final = {}
    room = room_for(effort)
    extra = {"cache_control": {"type": "ephemeral"}} if messages else {}

    def words_as_they_come():
        with client.beta.messages.stream(
            model=model,
            max_tokens=room,
            system=instructions,
            messages=messages or [{"role": "user", "content": prompt}],
            output_config={"effort": effort},  # how hard Claude thinks first
            # If Claude's safety checks decline a request, let Anthropic
            # quietly retry it on another suitable Claude model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            **extra,
        ) as stream:
            yield from stream.text_stream
            final["message"] = stream.get_final_message()

    try:
        with st.container(border=True):
            text = st.write_stream(words_as_they_come())
    except anthropic.AuthenticationError:
        st.error("The API key was not accepted. Please check it (see README.md).")
        return None
    except anthropic.RateLimitError:
        st.error("Too many requests right now. Please wait a minute and try again.")
        return None
    except anthropic.APIStatusError as e:
        st.error(f"Claude returned an error ({e.status_code}): {e.message}")
        return None
    except anthropic.APIConnectionError:
        st.error("Could not reach Anthropic. Please check your internet connection.")
        return None

    msg = final.get("message")
    if msg is None:
        return None
    if msg.stop_reason == "refusal":
        st.error("Claude declined this request. Try wording it differently.")
        return None
    note = usage_note(msg, effort, model)
    if msg.stop_reason == "max_tokens":
        note += " It ran out of space and stopped early - try a shorter length or a lower effort."
    return (text if isinstance(text, str) else "".join(map(str, text))), note


# Your own standing rules. Claude follows them EVERY time (answers and essays).
# They live in writing_rules.md, so you can change them in Notepad.
# (These are only used to make that file the first time.)
DEFAULT_WRITING_RULES = """\
- Do not simply combine, summarize, or copy and paste sections from the source \
material. Rewrite and reorganize it so that it reads as one coherent piece, \
written from the beginning as a unified whole.
- Do not introduce new ideas that are not supported by the source material.
- Take only the ideas and the worldview, never the language. Write in your \
own words and your own style: no copied or lightly reworded sentences, \
phrases, images or endings from the source material.
- Do not mention or refer to the existence of the source material: no "the \
essays say", "the author wrote", "as discussed elsewhere", no essay titles, no \
piece numbers.
- The final answer or essay must read as a complete, independent work that \
needs no knowledge of, or reference to, any other article.
"""


def load_writing_rules() -> str:
    """Your standing rules from writing_rules.md (made with the default rules
    the first time)."""
    if not WRITING_RULES_FILE.exists():
        WRITING_RULES_FILE.write_text(DEFAULT_WRITING_RULES, encoding="utf-8")
    return WRITING_RULES_FILE.read_text(encoding="utf-8-sig").strip()


def make_instructions(base: str) -> str:
    """Put together the basic instructions and YOUR standing rules."""
    text = base
    rules = load_writing_rules()
    if rules:
        text += ("\n\nThe author's standing rules. Always follow them. They come "
                 "before anything above:\n" + rules)
    return text


def rules_note():
    """A small fold-out showing the standing rules Claude always follows."""
    with st.expander("Your writing rules (Claude always follows these)"):
        st.markdown(load_writing_rules())
        st.caption("Change them in tab 5, or in writing_rules.md with Notepad.")


# ---------------------------------------------------------------------------
# THE WORLDVIEW GUIDE: Claude reads ALL the essays once and writes a summary
# of the author's way of seeing things. The chat uses it with every message.
# ---------------------------------------------------------------------------

WORLDVIEW_BUILD_INSTRUCTIONS = """You will get all the essays of one author, \
who writes on spirituality, nonduality and Hinduism.

Write a "worldview guide" for this author. Another AI will use it to answer \
new questions and write new essays the way this author would think - \
including on topics the author never wrote about. So it must capture HOW \
the author thinks, not just what each essay says.

Use these sections, with '## ' before each heading:
## 1. Core beliefs
The author's main beliefs, as clear numbered statements.
## 2. Key words and what the author means by them
Words the author uses in their own way (for example: self, karma, doer, \
freedom, Brahman, maya, ego, witness - whichever matter to this author).
## 3. How the ideas connect
The reasoning that links the beliefs: what follows from what.
## 4. What the author argues against
Views and common ideas the author rejects or corrects, and why.
## 5. Sources and examples the author uses
Scriptures, teachers, stories, images and everyday examples the author \
returns to.
## 6. Tensions and open questions
Places where the author holds a paradox, or where the essays differ or \
leave things open. Do not smooth these over.
## 7. Writing style
Tone, sentence length, how essays open and end, typical moves and phrases.
## 8. How this author would approach a new question
Practical guidance: the questions they would ask first, what they would \
look for, the conclusions they tend to reach.

Rules:
- Only describe what is really in the essays. Do not add your own views or \
outside teachings.
- After important points, name one or two essays where they appear, like \
(see: "What is Karma").
- Use simple, clear English. About 2,500 to 4,000 words.
- Start with '# Worldview guide' and one short paragraph saying what the \
guide is."""


def all_essays_text(pieces: pd.DataFrame) -> str:
    """Put every essay back together from its pieces (in order), one after another."""
    parts = []
    for essay, group in pieces.groupby("essay", sort=True):
        body = "\n\n".join(group.sort_values("piece")["text"])
        parts.append(f"=== Essay: {essay} ===\n\n{body}")
    return "\n\n\n".join(parts)


def load_worldview() -> str | None:
    """The saved worldview guide, or None if it has not been made yet."""
    if WORLDVIEW_FILE.exists():
        text = WORLDVIEW_FILE.read_text(encoding="utf-8-sig").strip()
        return text or None
    return None


def build_worldview(pieces: pd.DataFrame, fingerprint: str,
                    model: str = CLAUDE_MODEL, effort: str = "high") -> bool:
    """Send ALL essays to Claude, let it write the worldview guide, and save it.
    The guide that was there before is kept as worldview_old.md."""
    essays = all_essays_text(pieces)
    result = ask_claude(WORLDVIEW_BUILD_INSTRUCTIONS, f"The essays:\n\n{essays}",
                        effort=effort, model=model)
    if not result:
        return False
    new_guide = result[0].strip()
    if WORLDVIEW_FILE.exists():
        old_guide = WORLDVIEW_FILE.read_text(encoding="utf-8-sig")
        WORLDVIEW_OLD_FILE.write_text(old_guide, encoding="utf-8")
        # Keep section 9 (the points you approved after chats) - never lose them.
        kept = learned_section(old_guide)
        if kept and LEARNED_HEADING not in new_guide:
            new_guide += "\n\n" + kept
    WORLDVIEW_FILE.write_text(new_guide + "\n", encoding="utf-8")
    import datetime
    WORLDVIEW_INFO_FILE.write_text(json.dumps({
        "made on": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "essays": int(pieces["essay"].nunique()),
        "fingerprint": fingerprint,
        "note": result[1],
    }, indent=2))
    return True


def essay_to_word(essay: str, sources: pd.DataFrame, markers: bool = False) -> bytes:
    """Turn the essay text into a Word (.docx) file, with a list of sources at
    the end, so you can open and edit it in Word."""
    import io
    import re
    doc = Document()
    for line in essay.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            doc.add_heading(line[2:], level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=1)
        else:
            para = doc.add_paragraph()
            # **bold** parts become bold in Word; everything else is normal.
            for i, part in enumerate(re.split(r"\*\*", line)):
                if part:
                    para.add_run(part).bold = (i % 2 == 1)
    # The Sources list is added only for a checking draft (with [number] markers).
    # A clean version is a stand-alone piece: no sources, no mention of other articles.
    if markers:
        doc.add_heading("Sources", level=1)
        for n, r in enumerate(sources.itertuples(), start=1):
            doc.add_paragraph(f"[{n}] {r.essay} (piece {r.piece})")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CHAT: talk with Claude in sessions (tab 4).
# A session is remembered from the first message to the last: every time you
# send a message, Claude gets the WHOLE session again.
# Sessions are saved in the `store` folder in the same "vector" form as the
# essays:
#   - YOUR messages are cut into pieces and turned into meaning numbers, so a
#     later chat can find what you said before.
#   - Claude's replies are kept only so a session can be shown and continued.
#     They are never searched, so Claude's words never become "your" words.
# ---------------------------------------------------------------------------

CHAT_INSTRUCTIONS = """You are a writing partner for one author, who writes \
on spirituality, nonduality and Hinduism. You talk with the author in a \
conversation that can go on for a long time. The author may ask questions, \
ask for an essay, ask you to change something you wrote, or tell you what \
they think.

You get (together, your source material):
1. The author's worldview guide (at the end of these instructions). It is the \
heart of how this author thinks. Section 9, if there is one, holds points the \
author approved later. Where it differs from the sections before it, section \
9 wins.
2. With each message: numbered pieces of the author's writing that are \
closest to that message, and sometimes things the author said in earlier \
conversations (in their own words).

How to reply:
- Always think from the author's worldview, in every reply. When a reply \
touches anything the guide speaks about, it must agree with the guide. If a \
topic is new, apply the author's ideas to it.
- Every idea must be supported by the source material. You may use fresh \
examples and images to explain an idea, but do not add new claims or views.
- Remember the whole conversation. When the author asks for changes, change \
what you wrote before. Do not start from nothing unless asked.
- When the author tells you what they think, take it as their view for the \
rest of this conversation.
- If the worldview does not settle a question, say so briefly in your own words.
- Questions: answer in simple, clear English.
- Essays (when asked): a title, a short opening, a few sections with \
headings, and an ending. Format: '# ' before the title, '## ' before each \
section heading. No other markdown except **bold** for emphasis.
- Never mention the source material: no "the guide says", "your essays", \
essay titles or piece numbers.

Your own language (very important):
- Take the author's IDEAS, never their WORDS. The pieces and the guide show \
you what the author thinks. They are not text to reuse.
- Write every sentence fresh, in your own words and your own way of saying \
things. Do not copy or lightly reword the author's sentences, phrases, \
images, examples, jokes, openings or endings, and do not reuse the quotes in \
the guide.
- Do not imitate the author's style or "signature phrases". Find your own \
way to say each thing, and your own examples where an example helps.
- Do not repeat yourself either: when you write about the same idea again \
(in this conversation or in a new version), say it in new words.
- Exact wording may be kept only for scripture verses (with their \
reference) and for the author's key words, which you may use as terms \
(for example Brahman, maya, Sakshi), but not the sentences around them."""

# Claude reads a session and suggests what could be added to the worldview.
WORLDVIEW_SUGGEST_INSTRUCTIONS = """You help keep one author's worldview \
guide up to date. The author writes on spirituality, nonduality and Hinduism.

You get the current guide, then a conversation between the AUTHOR and an AI \
writing ASSISTANT. Find what the conversation shows about the author's views \
that the guide does not already say: new beliefs, sharper or corrected \
beliefs, the meaning of a key word, what they argue against, or how they \
like to write.

Rules:
- Use only what the AUTHOR said, corrected, or clearly agreed to. The \
assistant's words are NOT evidence. Use an assistant's point only if the \
author clearly agreed with that exact point, and then quote the author's \
agreement.
- Do not suggest anything the guide already says.
- Write each point as one or two clear sentences, in the guide's style, as a \
statement about the author's view (for example: "The author holds that...").
- For each point, give the guide section it belongs to (its name, for \
example "Core beliefs" or "Writing style") and quote the author's own words \
that support it.
- If there is nothing new, give an empty list. Fewer, solid points are better \
than many weak ones."""

WORLDVIEW_SUGGEST_SCHEMA = {
    "type": "object",
    "properties": {
        "points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "point": {"type": "string"},
                    "author_words": {"type": "string"},
                },
                "required": ["section", "point", "author_words"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["points"],
    "additionalProperties": False,
}

CHAT_SESSION_COLS = ["session", "title", "created", "updated", "model", "effort", "reviewed"]
CHAT_MESSAGE_COLS = ["session", "turn", "role", "text", "sent", "time", "model", "effort",
                     "note", "pieces", "past"]
CHAT_PIECE_COLS = ["session", "turn", "kind", "text", "time"]


def _now():
    import datetime
    return datetime.datetime.now()


def _read_table(path: Path, cols: list[str]) -> pd.DataFrame:
    return pd.read_pickle(path) if path.exists() else pd.DataFrame(columns=cols)


def _write_table(df: pd.DataFrame, path: Path):
    """Save safely: write a new file first, then swap it in (so a crash in
    the middle can never leave a half-written file)."""
    STORE_FOLDER.mkdir(exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.reset_index(drop=True).to_pickle(tmp, compression=None)
    tmp.replace(path)


def _add_rows(df: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows, columns=df.columns)
    return pd.concat([df, new], ignore_index=True) if len(df) else new


def load_sessions() -> pd.DataFrame:
    """One row per session."""
    return _read_table(CHAT_SESSIONS_FILE, CHAT_SESSION_COLS)


def load_messages(session: str | None = None) -> pd.DataFrame:
    """All messages, or only the messages of one session, in order."""
    m = _read_table(CHAT_MESSAGES_FILE, CHAT_MESSAGE_COLS)
    if session is None:
        return m
    m = m[m["session"] == session]
    return (m.assign(_order=(m["role"] == "claude").astype(int))
             .sort_values(["turn", "_order"]).drop(columns="_order"))


def _save_chat_vectors(pcs: pd.DataFrame, nums: np.ndarray):
    _write_table(pcs, CHAT_PIECES_FILE)
    tmp = CHAT_NUMBERS_FILE.with_name("chat_numbers.tmp.npy")
    np.save(tmp, nums)
    tmp.replace(CHAT_NUMBERS_FILE)
    CHAT_INFO_FILE.write_text(json.dumps({"model": MODEL_NAME}))


def load_chat_vectors() -> tuple[pd.DataFrame, np.ndarray]:
    """Your messages (and the worldview points you approved), cut into
    pieces, with their meaning numbers."""
    pcs = _read_table(CHAT_PIECES_FILE, CHAT_PIECE_COLS)
    if not len(pcs):
        return pcs, np.zeros((0, 384), dtype=np.float32)
    info = json.loads(CHAT_INFO_FILE.read_text()) if CHAT_INFO_FILE.exists() else {}
    nums = np.load(CHAT_NUMBERS_FILE) if CHAT_NUMBERS_FILE.exists() else np.zeros((0, 384))
    if len(nums) != len(pcs) or info.get("model") != MODEL_NAME:
        # The numbers don't fit (for example, the meaning model was changed).
        # The text is still here, so just make the numbers again.
        nums = to_numbers(load_model(), pcs["text"].tolist())
        _save_chat_vectors(pcs, nums)
    return pcs, nums


def add_to_chat_vectors(session: str, turn: int, kind: str, text: str, time: str):
    """Cut a text into pieces (like the essays), turn them into meaning
    numbers, and add them to the store. kind = "you" or "approved"."""
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    texts = cut_into_pieces(paragraphs, WORDS_PER_PIECE) if paragraphs else []
    if not texts:
        return
    pcs, nums = load_chat_vectors()
    pcs = _add_rows(pcs, [{"session": session, "turn": turn, "kind": kind,
                           "text": t, "time": time} for t in texts])
    new_nums = to_numbers(load_model(), texts)
    nums = np.vstack([nums, new_nums]) if len(nums) else new_nums
    _save_chat_vectors(pcs, nums)


def _short(text: str, n: int = 70) -> str:
    line = " ".join(text.split())
    return line if len(line) <= n else line[:n - 3].rstrip() + "..."


def _file_name(reply: str) -> str:
    """A Word file name from the first line of a reply (its title)."""
    first = next((line for line in reply.splitlines() if line.strip()), "reply")
    safe = "".join(ch for ch in first if ch.isalnum() or ch in " -_").strip()[:60]
    return f"{safe or 'reply'}.docx"


def new_session_id() -> str:
    sid = f"{_now():%Y-%m-%d_%H%M%S}"
    existing = set(load_sessions()["session"])
    while sid in existing:
        sid += "_x"
    return sid


def save_round(session: str, you_text: str, sent: str, reply: str, note: str,
               model: str, effort: str, pieces_used: list, past_used: list,
               time: str | None = None, title: str | None = None):
    """Save one round (your message + Claude's reply) straight away, so
    nothing is lost if the page is refreshed."""
    time = time or f"{_now():%Y-%m-%d %H:%M}"
    sessions, msgs = load_sessions(), load_messages()
    mine = msgs["session"] == session
    turn = int(msgs.loc[mine, "turn"].max()) + 1 if mine.any() else 1
    msgs = _add_rows(msgs, [
        {"session": session, "turn": turn, "role": "you", "text": you_text, "sent": sent,
         "time": time, "model": model, "effort": effort, "note": "", "pieces": [], "past": []},
        {"session": session, "turn": turn, "role": "claude", "text": reply, "sent": "",
         "time": time, "model": model, "effort": effort, "note": note,
         "pieces": pieces_used, "past": past_used}])
    if (sessions["session"] == session).any():
        i = sessions.index[sessions["session"] == session][0]
        sessions.loc[i, ["updated", "model", "effort"]] = [time, model, effort]
    else:
        sessions = _add_rows(sessions, [{
            "session": session, "title": title or _short(you_text), "created": time,
            "updated": time, "model": model, "effort": effort, "reviewed": 0}])
    _write_table(msgs, CHAT_MESSAGES_FILE)
    _write_table(sessions, CHAT_SESSIONS_FILE)
    add_to_chat_vectors(session, turn, "you", you_text, time)


def delete_session(session: str):
    """Delete a session: its messages and the numbers of your messages.
    (Worldview points you approved from it stay - they are in your guide.)"""
    sessions, msgs = load_sessions(), load_messages()
    _write_table(sessions[sessions["session"] != session], CHAT_SESSIONS_FILE)
    _write_table(msgs[msgs["session"] != session], CHAT_MESSAGES_FILE)
    pcs, nums = load_chat_vectors()
    if len(pcs):
        keep = ~((pcs["session"] == session) & (pcs["kind"] == "you")).values
        _save_chat_vectors(pcs[keep], nums[keep])


def mark_reviewed(session: str, upto: int):
    sessions = load_sessions()
    sessions.loc[sessions["session"] == session, "reviewed"] = upto
    _write_table(sessions, CHAT_SESSIONS_FILE)


def find_for_message(query: str, session: str, pieces: pd.DataFrame,
                     vecs: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The essay pieces closest to a message, and the closest things you said
    in OTHER sessions (or worldview points you approved)."""
    q = to_numbers(load_model(), [query])[0]
    top = (pieces.assign(score=vecs @ q)
           .sort_values("score", ascending=False).head(CHAT_TOP_K))
    pcs, nums = load_chat_vectors()
    past = pcs.iloc[0:0].assign(score=[])
    if len(pcs):
        other = ~((pcs["session"] == session) & (pcs["kind"] == "you")).values
        past = pcs[other].assign(score=nums[other] @ q)
        past = (past[past["score"] >= PAST_MIN_SCORE]
                .sort_values("score", ascending=False).head(PAST_TOP_K))
    return top, past


def message_for_claude(text: str, top: pd.DataFrame, past: pd.DataFrame) -> str:
    """Your message, as Claude gets it: with the pieces found for it."""
    parts = [f"Pieces of the author's writing, closest to this message. Use them "
             f"for their IDEAS only. Do not reuse their wording:\n\n"
             f"{number_the_pieces(top)}"]
    if len(past):
        parts.append("Things the author said in earlier conversations (their own "
                     "words), or points they approved for their worldview:\n\n" +
                     "\n\n".join(f"- ({r.time}) {r.text}" for r in past.itertuples()))
    parts.append(f"---\n\nThe author's message:\n{text}")
    return "\n\n".join(parts)


def without_style_section(guide: str) -> str:
    """The guide without its "Writing style" section. That section is about
    the author's LANGUAGE (phrases, openings, endings), and Claude should
    take only the worldview, so it never gets it. (The file is not changed.)"""
    out, skip = [], False
    for line in guide.splitlines():
        if line.startswith("## "):
            skip = "writing style" in line.lower()
        if not skip:
            out.append(line)
    return "\n".join(out)


def chat_system(guide: str) -> str:
    """Everything Claude is told in the chat: how to reply, YOUR writing
    rules, and the worldview guide (read fresh for every message, without its
    "Writing style" section)."""
    return (make_instructions(CHAT_INSTRUCTIONS) +
            "\n\n---\n\nThe author's worldview guide (ideas only - never reuse "
            "its wording or quotes):\n\n" + without_style_section(guide))


# --- The "own words" check: does a reply copy wording from the essays? ---------
COPY_RUN_WORDS = 7  # this many words in a row, the same as in an essay = copied


def _words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower().replace("\u2019", "'").replace("'", ""))


@st.cache_data(show_spinner=False)
def essay_word_runs(fingerprint: str) -> frozenset:
    """Every run of COPY_RUN_WORDS words in all your essays (made once)."""
    runs = set()
    for text in pieces["text"]:
        w = _words(text)
        runs.update(" ".join(w[i:i + COPY_RUN_WORDS]) for i in range(len(w) - COPY_RUN_WORDS + 1))
    return frozenset(runs)


def copied_parts(reply: str) -> list[str]:
    """The parts of a reply that are word for word the same as an essay
    (COPY_RUN_WORDS words in a row or more), joined into longer stretches."""
    runs = essay_word_runs(fingerprint)
    w = _words(reply)
    hit = [False] * len(w)
    for i in range(len(w) - COPY_RUN_WORDS + 1):
        if " ".join(w[i:i + COPY_RUN_WORDS]) in runs:
            for j in range(i, i + COPY_RUN_WORDS):
                hit[j] = True
    parts, cur = [], []
    for word, h in zip(w, hit):
        if h:
            cur.append(word)
        elif cur:
            parts.append(" ".join(cur))
            cur = []
    if cur:
        parts.append(" ".join(cur))
    return parts


def conversation_for_claude(session: str | None, new_sent: str) -> list[dict]:
    """The WHOLE session so far, plus the new message. This is how Claude
    remembers everything from the beginning."""
    out = []
    if session:
        for r in load_messages(session).itertuples():
            if r.role == "you":
                out.append({"role": "user", "content": r.sent or r.text})
            else:
                out.append({"role": "assistant", "content": r.text or "(empty)"})
    out.append({"role": "user", "content": new_sent})
    return out


# --- Updating the worldview guide, only with your approval -------------------

LEARNED_HEADING = "## 9. Learned from my conversations"


def learned_section(guide: str) -> str:
    """Section 9 of the guide (the points you approved), or "" if none yet."""
    i = guide.find(LEARNED_HEADING)
    if i < 0:
        return ""
    j = guide.find("\n## ", i + len(LEARNED_HEADING))
    return guide[i:j if j >= 0 else len(guide)].strip()


def add_learned_points(points: list[tuple[str, str]], session: str, upto: int):
    """Add the points you approved to section 9 of the guide (the old guide
    is kept as worldview_old.md), and store them as meaning numbers too."""
    guide = load_worldview() or ""
    WORLDVIEW_OLD_FILE.write_text(guide, encoding="utf-8")
    today = f"{_now():%Y-%m-%d}"
    lines = "\n".join(f"- {text.strip()} (about: {section}; approved {today})"
                      for section, text in points)
    sec = learned_section(guide)
    if sec:
        guide = guide.replace(sec, sec + "\n" + lines, 1)
    else:
        guide = (guide.rstrip() + f"\n\n{LEARNED_HEADING}\n\nPoints the author approved "
                 f"after conversations. Where they differ from the sections above, "
                 f"these win.\n\n{lines}")
    WORLDVIEW_FILE.write_text(guide.strip() + "\n", encoding="utf-8")
    for _, text in points:
        add_to_chat_vectors(session, upto, "approved", text.strip(), f"{_now():%Y-%m-%d %H:%M}")


def suggest_worldview_points(session: str, model: str, effort: str):
    """Claude reads the session and suggests points for the worldview guide.
    Gives back (list of points, a cost note), or None if it failed.
    Nothing is changed here - you decide what goes in."""
    import anthropic
    guide = load_worldview() or ""
    row = load_sessions().set_index("session").loc[session]
    reviewed = int(row["reviewed"] or 0)
    transcript = "\n\n".join(
        f"[Round {r.turn}] {'AUTHOR' if r.role == 'you' else 'ASSISTANT'}:\n{r.text}"
        for r in load_messages(session).itertuples())
    prompt = (f"The current worldview guide:\n\n{guide}\n\n---\n\n"
              f"The conversation:\n\n{transcript}")
    if reviewed:
        prompt += (f"\n\n---\n\nRounds 1 to {reviewed} were already reviewed. Only suggest "
                   f"points from round {reviewed + 1} onward (use the earlier rounds only "
                   f"to understand them).")
    client = anthropic.Anthropic(api_key=get_claude_key())
    try:
        with st.spinner("Claude is reading the session..."):
            with client.beta.messages.stream(
                model=model, max_tokens=room_for(effort),
                system=WORLDVIEW_SUGGEST_INSTRUCTIONS,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": effort,
                               "format": {"type": "json_schema",
                                          "schema": WORLDVIEW_SUGGEST_SCHEMA}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ) as stream:
                msg = stream.get_final_message()
    except anthropic.APIError as e:
        st.error(f"Claude returned an error: {getattr(e, 'message', e)}")
        return None
    if msg.stop_reason in ("refusal", "max_tokens"):
        st.error("Claude could not finish this. Try again, or with a lower effort.")
        return None
    try:
        text = next(b.text for b in msg.content if b.type == "text")
        points = json.loads(text)["points"]
    except (StopIteration, ValueError, KeyError):
        st.error("Claude's answer could not be read. Please try again.")
        return None
    return points, usage_note(msg, effort, model)


# --- Moving the old `history` folder into the chat sessions (done once) -------

def move_old_history() -> int:
    """Turn every saved answer and essay from the old History tab into a chat
    session (versions of one essay become one session with several rounds).
    Then the folder is renamed history_moved_backup (nothing is deleted).
    Gives back how many sessions were made."""
    items = []
    for f in sorted(OLD_HISTORY_FOLDER.glob("*.json")):
        try:
            h = json.loads(f.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        h["_stem"] = f.stem
        items.append(h)
    groups: dict[str, list[dict]] = {}
    for h in items:
        groups.setdefault(h.get("group") or h["_stem"], []).append(h)
    existing = set(load_sessions()["session"])
    made = 0
    for gid, hs in groups.items():
        if gid in existing:
            continue  # already moved before
        hs.sort(key=lambda h: (h.get("version", 1), h["_stem"]))
        for k, h in enumerate(hs):
            ctx = (h.get("context") or "").strip()
            if h.get("kind") == "essay":
                ask = (f"Write an essay on: {h['prompt']}" if k == 0 else
                       "Write a new version of this essay.")
                if h.get("length"):
                    ask += f" Length: {h['length']}."
            else:
                ask = h["prompt"]
            you_text = ask + (f"\n\nContext: {ctx}" if ctx else "")
            sent = (h.get("sent") or {}).get("prompt") or you_text
            if sent.startswith("Worldview guide:") and "\n\n---\n\n" in sent:
                sent = sent.split("\n\n---\n\n", 1)[1]  # the guide now goes in the system part
            save_round(gid, you_text, sent, h.get("text", ""), h.get("note", ""),
                       h.get("model") or CLAUDE_MODEL, h.get("effort") or "high",
                       h.get("sources", []), [], time=h.get("time"),
                       title=_short(h["prompt"]))
        made += 1
    try:
        OLD_HISTORY_FOLDER.rename(OLD_HISTORY_FOLDER.with_name("history_moved_backup"))
    except OSError:
        pass
    return made


# ---------------------------------------------------------------------------
# The map drawing (used in tab 3)
# ---------------------------------------------------------------------------

def draw_map(pieces: pd.DataFrame, highlight=None, star=None, top=None):
    """Draw every piece as a dot. Each essay gets its own colour.
    If a question is given, draw it as a star with lines to the best pieces."""
    fig = go.Figure()
    for essay, group in pieces.groupby("essay"):
        hover = [f"<b>{essay}</b> - piece {p}<br>" + "<br>".join(_wrap(t[:300] + "..."))
                 for p, t in zip(group["piece"], group["text"])]
        dim = highlight is not None and essay != highlight
        fig.add_trace(go.Scatter(
            x=group["x"], y=group["y"], mode="markers", name=essay,
            marker=dict(size=9, opacity=0.15 if dim else 0.85),
            hovertext=hover, hoverinfo="text"))
    if star is not None:
        for _, row in top.iterrows():
            fig.add_trace(go.Scatter(
                x=[star[0], row["x"]], y=[star[1], row["y"]], mode="lines",
                line=dict(color="black", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=top["x"], y=top["y"], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="black", width=2))))
        fig.add_trace(go.Scatter(
            x=[star[0]], y=[star[1]], mode="markers", name="your question",
            marker=dict(symbol="star", size=24, color="gold",
                        line=dict(color="black", width=1.5))))
    fig.update_layout(height=700, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      legend=dict(font=dict(size=10)), margin=dict(l=0, r=0, t=10, b=0))
    return fig


def _wrap(text: str, width: int = 70) -> list[str]:
    """Break long text into short lines so the hover box is readable."""
    lines, line = [], ""
    for word in text.split():
        if len(line) + len(word) > width:
            lines.append(line)
            line = ""
        line += word + " "
    return lines + [line]


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Part 1 - Visualizer", layout="wide")
st.title("RAG Part 1 - see how your essays become a searchable library")

files = sorted(f for f in DOCS_FOLDER.glob("*.docx") if not f.name.startswith("~$"))
if not files:
    st.error(f"No .docx files found in {DOCS_FOLDER}")
    st.stop()

fingerprint = folder_fingerprint(files)
docs, pieces, vecs, report = build_library(fingerprint)

# Tell the user what was done this time.
done = report["new"] + report["changed"] + report["removed"]
if not done:
    st.info(f"Nothing changed since last time. All {len(report['reused'])} essays were "
            f"loaded from the saved library. No work needed.")
else:
    lines = [f"**Library updated.** Reused **{len(report['reused'])}** essays as they were."]
    if report["new"]:
        lines.append(f"New, processed now: {', '.join(report['new'])}")
    if report["changed"]:
        lines.append(f"Changed, processed again: {', '.join(report['changed'])}")
    if report["removed"]:
        lines.append(f"Removed from the library: {', '.join(report['removed'])}")
    st.success("\n\n".join(lines))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Essays found", len(docs))
c2.metric("Essays used", int((docs["status"] == "used").sum()))
c3.metric("Pieces made", len(pieces))
c4.metric("Numbers per piece", vecs.shape[1])

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Documents", "2. Pieces", "3. Meaning map", "4. Chat", "5. My worldview & rules"])

# --- Tab 1: STEP 1 - READ ---------------------------------------------------
with tab1:
    st.markdown("**Step 1 - Read.** Each Word file is opened and only the plain text "
                "is kept. Pictures and formatting are thrown away. That is why a big "
                "file (lots of pictures) can still have only a few words.")
    st.dataframe(docs.drop(columns=["file key"]).sort_values("words", ascending=False),
                 width="stretch", hide_index=True, height=600)

# --- Tab 2: STEP 2 - CUT ----------------------------------------------------
with tab2:
    st.markdown(f"**Step 2 - Cut.** Each essay is cut into pieces of about "
                f"**{WORDS_PER_PIECE} words**. We never cut in the middle of a paragraph. "
                f"Each piece remembers which essay it came from.")
    used = docs[docs["status"] == "used"]["essay"].tolist()
    default = used.index("What is Karma") if "What is Karma" in used else 0
    chosen = st.selectbox("Pick an essay", used, index=default)
    colours = ["#e3f2fd", "#fff3e0", "#e8f5e9", "#fce4ec", "#ede7f6", "#e0f7fa"]
    for i, row in enumerate(pieces[pieces["essay"] == chosen].itertuples()):
        st.markdown(
            f"<div style='background:{colours[i % len(colours)]};color:#222;padding:12px;"
            f"border-radius:8px;margin-bottom:10px'><b>Piece {row.piece}</b> "
            f"<small>({row.words} words)</small><br>{row.text.replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True)

# --- Tab 3: STEP 3 + 4 - NUMBERS and the MAP --------------------------------
with tab3:
    st.markdown(f"**Step 3 - Numbers.** Every piece was turned into "
                f"**{vecs.shape[1]} numbers** that stand for its meaning. "
                f"**Step 4 - Store.** These were saved in the `store` folder.\n\n"
                f"To draw them, we squeeze the {vecs.shape[1]} numbers down to 2 (left-right "
                f"and up-down). **Each dot is one piece. Dots close together mean similar "
                f"things.** Each essay has its own colour. Hover over a dot to read it.")
    focus = st.selectbox("Highlight one essay (optional)", ["(none)"] + sorted(pieces["essay"].unique()))
    st.plotly_chart(draw_map(pieces, highlight=None if focus == "(none)" else focus),
                    width="stretch")
    with st.expander("See the actual numbers for one piece"):
        st.write(f"First piece of '{pieces.iloc[0]['essay']}' - first 20 of {vecs.shape[1]} numbers:")
        st.code(np.array2string(vecs[0][:20], precision=3, separator=", "))

# --- Tab 4: CHAT - talk with Claude in sessions --------------------------------
def _open_session(sid: str):
    st.session_state["chat_sid"] = sid
    st.session_state.pop("wv_suggest", None)
    s = load_sessions().set_index("session")
    if sid in s.index:  # use the model and effort this session used last
        if s.loc[sid, "model"] in MODELS:
            st.session_state["chat_model"] = s.loc[sid, "model"]
        if s.loc[sid, "effort"] in EFFORTS:
            st.session_state["chat_effort"] = s.loc[sid, "effort"]


def _new_session():
    st.session_state["chat_sid"] = None
    st.session_state.pop("wv_suggest", None)


def _delete_session(sid: str):
    delete_session(sid)
    _new_session()
    st.session_state["chat_msg"] = "The session was deleted."


with tab4:
    if OLD_HISTORY_FOLDER.exists():
        with st.spinner("Moving your saved answers and essays into chat sessions..."):
            n_moved = move_old_history()
        st.success(f"Your {n_moved} saved items from the old History were moved into chat "
                   f"sessions (see the list on the left). The old folder was kept as "
                   f"`history_moved_backup`, just in case.")
    st.markdown("Talk with Claude. Ask questions, ask for an essay, ask for changes, or "
                "tell it what you think. **Claude remembers the whole session**, and it "
                "gets **your worldview guide** and your **writing rules** with every "
                "message. Everything is saved straight away.")
    if "chat_msg" in st.session_state:
        st.success(st.session_state.pop("chat_msg"))

    guide = load_worldview()
    sessions = load_sessions().sort_values("updated", ascending=False)
    all_msgs = load_messages()
    if st.session_state.get("chat_sid") not in set(sessions["session"]):
        st.session_state["chat_sid"] = None
    sid = st.session_state.get("chat_sid")

    left, right = st.columns([1, 3], gap="large")

    # --- Left: the list of sessions ---
    with left:
        st.button("New session", type="primary", width="stretch", on_click=_new_session)
        find = st.text_input("Search my sessions", key="chat_search")
        shown = 0
        for r in sessions.itertuples():
            mine = all_msgs[all_msgs["session"] == r.session]
            if find.strip() and find.lower() not in (
                    r.title + " " + " ".join(mine["text"])).lower():
                continue
            shown += 1
            st.button(f"{r.title}", key=f"open_{r.session}", width="stretch",
                      type="primary" if r.session == sid else "secondary",
                      help=f"Last used {r.updated} - {len(mine) // 2} rounds",
                      on_click=_open_session, args=(r.session,))
        if not shown:
            st.caption("No sessions yet." if not len(sessions) else "Nothing found.")

    # --- Right: the open session ---
    with right:
        msgs = load_messages(sid) if sid else load_messages().iloc[0:0]
        h1, h2 = st.columns([5, 1])
        h1.subheader(sessions.set_index("session").loc[sid, "title"] if sid else "New session")
        if sid:
            with h2.popover("Delete"):
                st.write("Delete this whole session for good? (Worldview points you "
                         "approved from it stay in your guide.)")
                st.button("Yes, delete it", key="chat_del",
                          on_click=_delete_session, args=(sid,))
        chat_model, chat_effort = choose_model("chat", "medium")
        if guide is None:
            st.error("The chat needs your **worldview guide** first. Make it in tab "
                     "**5. My worldview & rules**.")
        else:
            saved_on = _now().fromtimestamp(WORLDVIEW_FILE.stat().st_mtime)
            st.caption(f"Worldview guide: {len(guide.split()):,} words, last saved "
                       f"{saved_on:%Y-%m-%d %H:%M}. Claude gets it with every message.")
        if not has_claude_key():
            st.error("No Claude API key found. See **'Setting up the AI answer'** in README.md.")

        convo = st.container()
        with convo:
            if not len(msgs):
                st.info("Type your first message below to start a new session.")
            for r in msgs.itertuples():
                with st.chat_message("user" if r.role == "you" else "assistant"):
                    st.markdown(r.text)
                    if r.role == "claude":
                        st.caption(r.note)
                        copied = copied_parts(r.text)
                        if copied:
                            st.warning(f"**Copied wording:** {len(copied)} "
                                       f"part{'s' if len(copied) > 1 else ''} of this reply "
                                       f"match your essays word for word. Ask Claude to "
                                       f"\"say it in your own words\" if you don't want this.\n\n"
                                       + "\n".join(f"- \"...{c}...\"" for c in copied[:5]))
                        c1, c2 = st.columns([3, 1])
                        with c1.expander("Pieces used for this reply"):
                            for n, p in enumerate(r.pieces, start=1):
                                st.write(f"[{n}] {p['essay']} (piece {p['piece']})")
                            for p in r.past:
                                st.write(f"From what you said on {p['time']}: "
                                         f"\"{_short(p['text'], 120)}\"")
                        c2.download_button(
                            "Word file", data=essay_to_word(r.text, pd.DataFrame()),
                            file_name=_file_name(r.text),
                            mime="application/vnd.openxmlformats-officedocument."
                                 "wordprocessingml.document",
                            on_click="ignore", key=f"dl_{sid}_{r.turn}")

        st.caption("**Privacy:** when you press Enter, your message, the whole session so "
                   f"far, about {CHAT_TOP_K} pieces of your essays, your worldview guide "
                   "and your writing rules are sent to Anthropic. Nothing is sent before.")
        text = st.chat_input("Write a message", key="chat_box",
                             disabled=guide is None or not has_claude_key())
        if text:
            new_sid = sid or new_session_id()
            with convo:
                with st.chat_message("user"):
                    st.markdown(text)
                # Short follow-ups ("make it shorter") are searched together with
                # your previous message, so the right pieces are still found.
                before = msgs.loc[msgs["role"] == "you", "text"].tolist()
                query = text if len(text.split()) >= 25 or not before else before[-1] + "\n" + text
                top, past = find_for_message(query, new_sid, pieces, vecs)
                sent = message_for_claude(text, top, past)
                with st.chat_message("assistant"):
                    result = ask_claude(chat_system(load_worldview()), "", chat_effort,
                                        chat_model, messages=conversation_for_claude(sid, sent))
            if result:
                save_round(new_sid, text, sent, result[0], result[1], chat_model, chat_effort,
                           [{"essay": r.essay, "piece": int(r.piece)} for r in top.itertuples()],
                           [{"text": r.text, "time": r.time} for r in past.itertuples()])
                st.session_state["chat_sid"] = new_sid
                st.rerun()

        # --- Teach the worldview from this session (only what you approve) ---
        if sid and len(msgs):
            st.divider()
            st.markdown("#### Update my worldview from this session")
            reviewed = int(sessions.set_index("session").loc[sid, "reviewed"] or 0)
            last = int(msgs["turn"].max())
            st.caption(f"Rounds checked so far: {reviewed} of {last}. Claude suggests "
                       f"points from **your** messages only. **Nothing changes your guide "
                       f"until you approve it.** (Uses the model and effort chosen above.)")
            sug = st.session_state.get("wv_suggest")
            if sug and sug["sid"] != sid:
                sug = None
            if not sug:
                if reviewed >= last:
                    st.caption("Every round of this session has been checked already.")
                elif st.button("Suggest worldview updates from this session",
                               disabled=guide is None or not has_claude_key()):
                    out = suggest_worldview_points(sid, chat_model, chat_effort)
                    if out is not None:
                        st.session_state["wv_suggest"] = {
                            "sid": sid, "upto": last, "points": out[0], "note": out[1],
                            "id": f"{_now():%H%M%S%f}"}
                        st.rerun()
            else:
                k = sug["id"]
                if not sug["points"]:
                    st.info("Claude found nothing new for your worldview in this session.")
                for i, p in enumerate(sug["points"]):
                    with st.container(border=True):
                        st.checkbox("Approve this point", key=f"wv_ok_{k}_{i}")
                        st.text_area("The point (you can change the words)", p["point"],
                                     key=f"wv_txt_{k}_{i}", height=80)
                        st.caption(f"Section: {p['section']}.  Your words: "
                                   f"\"{_short(p['author_words'], 300)}\"")
                st.caption(sug["note"])
                b1, b2 = st.columns(2)
                if b1.button("Save the ones I approved" if sug["points"] else
                             "OK, mark as checked", type="primary"):
                    chosen = [(p["section"], st.session_state.get(f"wv_txt_{k}_{i}", p["point"]))
                              for i, p in enumerate(sug["points"])
                              if st.session_state.get(f"wv_ok_{k}_{i}")]
                    if chosen:
                        add_learned_points(chosen, sid, sug["upto"])
                    mark_reviewed(sid, sug["upto"])
                    st.session_state.pop("wv_suggest", None)
                    st.session_state["chat_msg"] = (
                        f"{len(chosen)} point{'s' if len(chosen) != 1 else ''} added to your "
                        f"worldview guide (section 9). The old guide was kept as "
                        f"worldview_old.md." if chosen else
                        "Nothing was added to your worldview guide.")
                    st.rerun()
                if b2.button("Cancel (decide later)"):
                    st.session_state.pop("wv_suggest", None)
                    st.rerun()


# --- Tab 5: THE WORLDVIEW GUIDE - used by the chat with every message ----------
with tab5:
    st.markdown(
        "Claude reads **all your essays at once** and writes a **worldview guide**: your "
        "core beliefs, the key words you use and what you mean by them, how your ideas "
        "connect, what you argue against, and your writing style.\n\n"
        "In the **chat** (tab 4), Claude gets this guide with **every message**, so it "
        "can answer new questions and write on new topics **the way you think**.\n\n"
        "**Section 9** holds the points you approved after chat sessions. Making the "
        "guide again keeps section 9.\n\n"
        "**You are the judge.** Read the guide. If something is wrong or missing, fix it "
        "below. Claude follows whatever the guide says.")

    # A message from the last click (for example "the guide was made").
    if "worldview_msg" in st.session_state:
        st.success(st.session_state.pop("worldview_msg"))

    guide_now = load_worldview()
    info = (json.loads(WORLDVIEW_INFO_FILE.read_text())
            if WORLDVIEW_INFO_FILE.exists() else {})

    if guide_now is None:
        st.info("No worldview guide yet. Click the button below to make it.")
    else:
        st.markdown(f"**Guide made on:** {info.get('made on', 'unknown')}, "
                    f"from **{info.get('essays', '?')} essays**. Saved in `worldview.md`.")
        if info.get("fingerprint") and info["fingerprint"] != fingerprint:
            st.warning("Your essays have changed since this guide was made (added, edited "
                       "or deleted). If the changes are big, make the guide again.")

    # --- Make (or make again) the guide ---
    total_words = int(pieces["words"].sum())
    with st.expander("Make the guide" if guide_now is None else "Make the guide again",
                     expanded=guide_now is None):
        guide_model, guide_effort = choose_model("guide", "high")
        # Estimate: these essays use about 2 tokens per word; the guide is ~12,000 tokens.
        guess = cost_in_dollars(guide_model, total_words * 2, 12000)
        st.warning(f"**Privacy and cost:** when you click, **all {pieces['essay'].nunique()} "
                   f"essays** (about {total_words:,} words) are sent to Anthropic. With this "
                   f"model it costs about **{guess:.2f} US dollars** (more at higher effort) "
                   f"and takes a few minutes.")
        if guide_now is not None:
            st.markdown("Making it again **replaces** the current guide, including any "
                        "changes you made (but **section 9**, the points you approved, is "
                        "kept). The current one is kept as `worldview_old.md`, just in case.")
        if not has_claude_key():
            st.error("No Claude API key found. See **'Setting up the AI answer'** in README.md.")
        elif st.button("Make the worldview guide", type="primary"):
            if build_worldview(pieces, fingerprint, guide_model, guide_effort):
                st.session_state["worldview_msg"] = (
                    "The worldview guide was made and saved. Read it below and fix "
                    "anything that is not right.")
                st.rerun()

    # --- Your standing writing rules ---
    st.subheader("Your writing rules")
    st.markdown("Claude follows these rules in **every chat reply** (answers and essays). "
                "Change them here (or in `writing_rules.md` with Notepad).")
    rules_now = load_writing_rules()
    with st.expander("See or change the rules", expanded=False):
        rules_key = f"rules_edit_{WRITING_RULES_FILE.stat().st_mtime_ns}"
        new_rules = st.text_area("One rule per line, starting with '- '.", rules_now,
                                 height=220, key=rules_key)
        if st.button("Save the rules"):
            WRITING_RULES_FILE.write_text(new_rules.strip() + "\n", encoding="utf-8")
            st.session_state["worldview_msg"] = "Your writing rules were saved."
            st.rerun()
        st.markdown(rules_now)
    st.subheader("Your worldview guide")

    # --- Edit the guide ---
    if guide_now is not None:
        with st.expander("Edit the guide"):
            # The key includes the file's save time, so after a rebuild the box
            # shows the new guide, not the old text.
            edit_key = f"guide_edit_{WORLDVIEW_FILE.stat().st_mtime_ns}"
            edited = st.text_area("Change anything, then click Save.", guide_now,
                                  height=500, key=edit_key)
            if st.button("Save my changes"):
                WORLDVIEW_FILE.write_text(edited.strip() + "\n", encoding="utf-8")
                st.session_state["worldview_msg"] = "Your changes to the guide were saved."
                st.rerun()

        # --- Show the guide ---
        with st.container(border=True):
            st.markdown(guide_now)
