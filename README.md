# RAG Part 1 – Visualizer

This is a small app that **shows you, with pictures, what "Part 1" of a RAG pipeline does** with your essays in the `new` folder.

It opens in your web browser. It **only reads** the `new` folder and **never changes** your essays.

---

## How to start it

**Easy way:** double-click **`run.bat`** in this folder.

**Or** in a terminal:

```
cd rag_viz
.venv\Scripts\streamlit run app.py
```

Your browser opens at **http://localhost:8501**.
To stop it, close the black terminal window (or press `Ctrl + C` in it).

### Setting it up on a new computer (first time only)

1. Install **Python 3.12** from python.org.
2. Put your essays (Word `.docx` files) in a folder called **`new`**, **next to** the `rag_viz` folder:
   ```
   some folder\
     new\          <- your essays
     rag_viz\      <- this app
   ```
3. In a terminal, inside the `rag_viz` folder, run:
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
4. For the chat, add a Claude API key (see **"Setting up the AI answer"** below).
5. Double-click `run.bat`.

Your own files (`claude_key.txt`, the `store` folder, `worldview.md`, `writing_rules.md`) are made on your computer and are **never uploaded** to GitHub. The `.gitignore` file makes sure of that.

---

## What is RAG? (one line)

RAG = **search first, then answer.** Before the AI answers a question, it first finds the right bits of your writing, then answers using them.

**Part 1** is the "getting ready" part: turning your essays into a library that can be searched by *meaning*. This app shows Part 1.

---

## The 4 steps of Part 1 (and where to see each one in the app)

| Step | What happens | In very simple words | See it in tab |
|---|---|---|---|
| 1. Read | Each `.docx` file is opened, only the plain text is kept | Take the words out, throw away pictures and fonts | **1. Documents** |
| 2. Cut | Each essay is cut into pieces of about 150 words | Big essays are hard to search, so cut them into small pieces | **2. Pieces** |
| 3. Numbers | Each piece is turned into 384 numbers | The numbers stand for the *meaning*. Similar meaning → similar numbers | **3. Meaning map** |
| 4. Store | Pieces + numbers are saved in the `store` folder | Save the work so it can be searched quickly later | (files in `store`) |

---

## The tabs in the app

Tabs 1–3 show how RAG works. Tab 4 (**Chat**) and tab 5 (**My worldview & rules**) are explained further down.

### 1. Documents
A table of all your essays: file size, how many words, how many pieces, and whether it was used.
A big file size does **not** mean a lot of words. Most of the size is usually pictures.

### 2. Pieces
Pick an essay and see how it was cut. Each coloured box is one piece.
We never cut in the middle of a paragraph.

### 3. Meaning map
**Every dot is one piece.** Each essay has its own colour.
**Dots close together talk about similar things**, even if they use different words.
Hover over a dot to read the piece. You can also highlight one essay.

> Note: each piece really has 384 numbers. A screen only has 2 directions (left–right, up–down), so we squeeze 384 down to 2 just for drawing. The map is a helpful picture, not perfect. Searching always uses all 384 numbers.

### 4. Chat
Talk with Claude about anything: ask questions, ask for essays, ask for changes. See **"Tab 4 – Chat"** below.

**Closeness scores** (used to find the right pieces for each message): higher = closer in meaning. For this model, about **0.6 or more** is a good match, and about **0.4 or less** is not related.

---

## Files in this folder

| File / folder | What it is |
|---|---|
| `app.py` | The app itself. Every part has simple comments. |
| `run.bat` | Double-click to start the app. |
| `README.md` | This guide. |
| `store/` | The saved "library" of essays **and your chat sessions**. ⚠️ The essay part is made again if deleted, but **your chat sessions are only here**, so don't delete the `chat_...` files. |
| `store/documents.csv` | One row per essay. |
| `store/pieces.csv` | One row per piece: essay name, piece number, text, and its spot on the map. |
| `store/numbers.npy` | The 384 meaning numbers for every piece. |
| `store/info.json` | Notes about how the library was built (model name, piece size). |
| `.venv/` | The Python tools this app needs. Kept separate so nothing else on your laptop is touched. |
| `claude_key.txt` | Your Claude API key, if you make this file. **Keep it secret.** |
| `worldview.md` | Your **worldview guide** (made in tab 5). You can open it in Notepad and edit it. It's kept outside `store`, so deleting `store` won't delete it. |
| `worldview_old.md` | The guide from before the last time it was made again, just in case. |
| `worldview_info.json` | When the guide was made, and from how many essays. |
| `writing_rules.md` | **Your writing rules.** Claude follows them in every reply. Edit in tab 5 or Notepad. |
| `store/chat_sessions.pkl` | One row per chat session (title, dates, model, effort, how many rounds were checked for the worldview). |
| `store/chat_messages.pkl` | Every chat message, in order: yours and Claude's, with the pieces used and the cost. |
| `store/chat_pieces.pkl` | **Your** chat messages (and the worldview points you approved), cut into pieces. |
| `store/chat_numbers.npy` | The 384 meaning numbers for each of those pieces. |
| `store/chat_info.json` | Which meaning model made those numbers. |
| `history_moved_backup/` | The old History files (answers and essays from before the chat). They were moved into chat sessions. Safe to delete. |
| `.streamlit/config.toml` | App settings: only this laptop can open it, no questions at start, no usage data sent. |

---

## When you add, edit or delete an essay

Just put the file in the `new` folder (or edit / delete it) and **refresh the browser page**.

The app **only works on what changed**. It does not redo everything.

| What you did | What the app does |
|---|---|
| Added a new essay | Reads, cuts and makes numbers for **that essay only** |
| Edited an essay | Does that **one essay again** |
| Deleted an essay | **Removes its pieces** from the library |
| Nothing | **Loads the saved library**, so it opens instantly |
| All other essays | **Reused** exactly as saved, with no work |

**How does it know what changed?** For every essay it saves a small "ID card": the file's **size** and **when it was last saved** (column `file key` in `store/documents.csv`). On refresh, it compares each file with its ID card. A different card means the file changed.

A green or blue box at the top of the app tells you what was done, for example:
> Library updated. Reused 46 essays as they were. New, processed now: My New Essay

**Two things are still redone every time something changes:**
- **The map.** The dots are arranged again from scratch, which takes only a few seconds. Groups may move to a different spot on the screen, but which pieces are close together stays about the same.
- **Everything, if you change a setting.** If you change `WORDS_PER_PIECE` or `MODEL_NAME`, old pieces and numbers no longer fit, so the whole library is rebuilt once.

**Only `.docx` files directly inside `new` are read.** PDFs, `.txt` files and sub-folders are ignored.

---

## Things you can change (at the top of `app.py`)

| Setting | Now | What it does |
|---|---|---|
| `WORDS_PER_PIECE` | 150 | Size of each piece. Smaller = more, sharper pieces. Bigger = fewer pieces with more context. |
| `CHAT_TOP_K` | 8 | How many essay pieces Claude gets with each chat message. |
| `PAST_TOP_K` | 3 | Up to how many things you said in **other** sessions are added to a message. |
| `PAST_MIN_SCORE` | 0.7 | How close those must be to the message to be added. Lower = more. |
| `COPY_RUN_WORDS` | 7 | How many words in a row, the same as in an essay, count as "copied wording". |
| `MODEL_NAME` | `BAAI/bge-small-en-v1.5` | The model that makes the meaning numbers. |
| `DOCS_FOLDER` | `..\new` | Which folder of essays to read. |

---

## The tools used (and why)

| Tool | Why |
|---|---|
| **python-docx** | Reads text out of Word files. |
| **fastembed** + **bge-small** model | Makes the meaning numbers. Free, runs **on your laptop**, so your essays stay private. (The model, about 100 MB, is downloaded once the first time.) |
| **t-SNE** (from scikit-learn) | Squeezes 384 numbers down to 2, to draw the map. |
| **Plotly** | Draws the map you can hover over and zoom. |
| **Streamlit** | Turns the Python script into a web page. |
| **anthropic** | Anthropic's official library for talking to Claude (step 5 only). |

---

## Tab 4 – Chat

One chat for everything: questions, essays, rewrites, and telling Claude what you think.

### Sessions
- On the left: **New session**, a **search** box, and the list of your sessions (newest first). Click one to open it and carry on.
- **Claude remembers the whole session**, from the first message to the last. Every time you send a message, Claude gets the whole session again. So you can say *"make it shorter"*, *"now write it for young professionals"* or *"no, I see karma differently: ..."*.
- Everything is **saved straight away**, after every reply. A refresh or closing the app loses nothing.
- **Delete** (top right of a session) removes the whole session, after asking you first.

### What Claude gets with every message
1. **Your worldview guide**, read fresh every time. So a point you approve is used from the very next message, in every session.
2. **Your writing rules.**
3. The **8 pieces of your essays** closest to your message. For short follow-ups like *"make it shorter"*, your previous message is searched together with it.
4. Up to **3 things you said in other sessions**, if they are close to the message (your own words only, **never Claude's replies**).
5. The **whole session so far**.

Below each reply: what it cost, **Pieces used for this reply**, and a **Word file** button.

### Your ideas, Claude's own words
Claude is told to take **only your worldview, never your language**:
- It must write every sentence **fresh, in its own words and its own style**: no copied or lightly reworded sentences, phrases, images, jokes, openings or endings from your essays.
- It doesn't repeat itself either. A new version says things in new words.
- It may keep only scripture verses (with their reference) and your key terms (Brahman, maya…).
- **Section 7 of your guide ("Writing style") is not sent to Claude.** It lists your signature phrases and endings, which made Claude copy them. It stays in `worldview.md`; the chat just skips it.
- The same rule is in your writing rules (tab 5), so you can see it and change it.

**The "Copied wording" check:** under every reply, the app compares it with all your essays. If **7 or more words in a row** are the same as in an essay, a yellow box shows those parts. A Gita verse quoted the same way can show up here too. That's expected. To fix the rest, just ask: *"say that in your own words."*

In a test, an essay on *Karma and freedom* used to match your essays for **12–14%** of its words. With the new rules it was about **3%**, and most of that was a Gita verse.

### How sessions are stored
Like the essays: **your messages** are cut into pieces and turned into **meaning numbers**, in the `store` folder. That's how a new chat can find what you said before. Claude's replies are kept too, so you can read and continue a session, but they are **never searched**. Claude's words never become "your" words.

### Updating your worldview (only what you approve)
Under each session there is **"Update my worldview from this session"**:
1. Click **Suggest worldview updates from this session**. Claude reads the session and suggests points. It may only use **what you said, corrected or clearly agreed to**, never its own ideas.
2. Each suggestion shows **your words** that support it. Tick **Approve this point** for the ones you agree with. You can change the wording first.
3. Click **Save the ones I approved**. Only those are added to your guide, in **section 9, "Learned from my conversations"**, with the date. Everything else is dropped.
4. The old guide is kept as `worldview_old.md`, just in case.

The app remembers which rounds were checked, so next time only the **new** rounds are looked at. It uses the model and effort chosen at the top of the chat.

### Your old History
Your saved answers and essays from before were **moved into chat sessions** (7 items became 6 sessions: the two versions of *Karma and freedom* became one session). The old files are kept in `history_moved_backup/`. You can delete that folder.

### Privacy and cost
When you press **Enter**, your message, the whole session so far, about 8 pieces of your essays, your worldview guide and your writing rules are sent to **Anthropic**. **Nothing is sent before.**

The guide is about 6,000 tokens, so every message costs a bit more than before. But the start of the session is **cached** at Anthropic, which makes re-reading it much cheaper. In a test with Haiku (low effort), message 1 cost 0.2 cents and message 2 cost 0.1 cents, with 14,046 of 17,299 tokens read from the cache. With Opus, expect about 3–10 US cents per message, more for long sessions and long essays.

## Your writing rules (built in)

Claude follows these rules in **every chat reply** (answers and essays):

- Don't simply combine, summarize, or copy and paste sections from the source material. Rewrite and reorganize it so it reads as **one coherent piece**, written from the beginning as a unified whole.
- Don't introduce **new ideas** that the source material doesn't support.
- **Don't mention or refer to the source material**: no "the essays say", "the author wrote", essay titles or piece numbers.
- The result must read as a **complete, independent work** that needs no other article.

They're kept in **`writing_rules.md`**. To change them, use tab **5. My worldview & rules → Your writing rules → See or change the rules**, or open the file in Notepad. Each line starting with `- ` is one rule. You can add your own, for example *"- Use British spelling."*

## Choosing the model and the effort

In tab 4 (**chat**) and tab 5 (**making the guide**) there are two pickers: **Model** and **Effort**. A session remembers the model and effort it used last.

**Model:** which Claude writes. Prices are per 1 million tokens (a token is about ¾ of a word). "Read" is your pieces and guide; "written" is Claude's thinking plus its text.

| Model | Good for | Read | Written |
|---|---|---|---|
| **Fable 5.1** | The most capable. Try it for your most important essays. | $10 | $50 |
| **Opus 5.5** (starts selected) | Very capable. The best everyday choice. | $4 | $20 |
| **Sonnet 5.5** | Good quality at half the price of Opus. | $2 | $10 |
| **Haiku 5.5** | Fast and very cheap. Fine for quick questions, weaker on deep ideas and style. | $0.10 | $0.50 |

**Effort:** how hard Claude thinks before it writes.

| Effort | When to use it |
|---|---|
| Low | Quick, simple questions |
| Medium | Most chat messages (starts selected in the chat) |
| High | Important essays, and the worldview guide (starts selected for the guide) |
| Extra high / Max | Hard or subtle topics. Slower, and costs more, because thinking is paid for as written tokens. |

**Every reply shows what it really cost**, for example:
> Written by claude-sonnet-5-5, effort high. It read 1,856 tokens and wrote 554 tokens, including its thinking. Cost: about 0.9 US cents.

This line is saved with the session, so you can compare models later.

In tests, the same question cost **0.1 cents** with Haiku (low), **0.9 cents** with Sonnet (high) and about **2 cents** with Opus (medium).

## Tab 5 – My worldview & rules

(The **writing rules** part is explained above, in "Your writing rules".)


**Why it's needed:** for one message, Claude sees only a few pieces of your writing, out of 647. Your worldview is spread across all 91 essays. The **worldview guide** gives Claude the **whole picture** in a few pages.

**Claude doesn't "learn" or get trained.** It **reads** the guide every time it writes, like a ghost-writer reading a briefing about you before starting.

**How the guide is made:** click **Make the worldview guide**. Claude reads **all your essays at once** (about 211,000 tokens) and writes a guide with 8 parts:
1. Core beliefs
2. Key words and what you mean by them
3. How the ideas connect
4. What you argue against
5. Sources and examples you use
6. **Tensions and open questions**: places where your essays disagree with each other (for example on free will)
7. Writing style
8. How you'd approach a new question

It costs about **1 US dollar** and takes a few minutes. The first one cost $1.09. Then the chat sends it with every message (see the chat's "Privacy and cost").

**Section 9 – Learned from my conversations:** the points you approved after chat sessions (see "Updating your worldview" above). Where they differ from sections 1–8, they win. **Making the guide again keeps section 9**, so approved points are never lost. You can also edit or delete them here, like the rest of the guide.

**You are the judge.** Read the guide. If something is wrong or missing, open **Edit the guide**, fix it, and click **Save my changes**. Or edit `worldview.md` in Notepad. Claude follows whatever the guide says. Section 6 is a good place to start: you can say which side of each tension you hold *now*.

**When to make it again:** after adding or changing many essays. The tab shows a yellow note when your essays have changed since the guide was made. Making it again **replaces** your edits, but the old guide is kept as `worldview_old.md`.

**Privacy:** making the guide sends **all your essays** to Anthropic, only when you click.

## Setting up the AI answer

(This is needed for the **chat** and for **making the worldview guide**.)

You need a **Claude API key**. It's like a password that lets the app use Claude.

1. Go to **https://console.anthropic.com**, sign in, and add some credit under **Billing**.
2. Go to **API Keys** and click **Create Key**. Copy it. It starts with `sk-ant-`.
3. Do **one** of these:
   - **Easy way:** in the `rag_viz` folder, make a file called **`claude_key.txt`** and paste the key in it (just the key, nothing else). Save it.
   - **Or** set it for your Windows account. In a terminal, run `setx ANTHROPIC_API_KEY "sk-ant-..."`, then close and reopen run.bat.
4. Refresh the app page. The chat box will work.

**Keep the key secret.** Don't share `claude_key.txt` or put it online. Anyone with the key can spend your credit.

### Settings for Claude (at the top of `app.py`)

| Setting | Now | What it does |
|---|---|---|
| `CLAUDE_MODEL` | `claude-opus-5-5` | Which model is selected when the app opens. |
| `MODELS` | 4 models | The models you can pick, and their prices (used to show the cost). |
| `EFFORTS` | low … max | The effort choices. |
| `CHAT_INSTRUCTIONS` | (a short text) | How Claude replies in the chat: always from your worldview, every idea supported, never mentioning the sources. |
| `WORLDVIEW_SUGGEST_INSTRUCTIONS` | (a short text) | How Claude suggests worldview points from a session (your words only). |
| `writing_rules.md` | 4 rules | Your standing rules (a file, not in `app.py`). |
| `WORLDVIEW_BUILD_INSTRUCTIONS` | (a short text) | What Claude is asked to put in the worldview guide. |

### If something goes wrong

| Message | What it means |
|---|---|
| "No Claude API key found" | No `claude_key.txt` and no `ANTHROPIC_API_KEY`. See the setup steps above. |
| "The API key was not accepted" | The key is wrong or was deleted. Make a new one. |
| "Too many requests" | Wait a minute and try again. |
| "Could not reach Anthropic" | Check your internet connection. |
| "Claude declined to answer" | Claude's safety checks said no. Try asking differently. |
| "The chat needs your worldview guide first" | Make the guide in tab 5. |
| "Claude's answer could not be read" | Rare, when suggesting worldview points. Click the button again. |
