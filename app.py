import os, re, time, requests, gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL    = "llama-3.3-70b-versatile"
LLM_API_URL  = "https://api.groq.com/openai/v1/chat/completions"

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "JobLens/1.0", "Accept": "application/json"})

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM = """\
You are JobLens, an expert job description analyst and career advisor.

Your task: given a job description, research the company and produce a structured analysis.

PROCESS:
1. Read the job description. Extract the company name and role title.
2. Use web_search 2–3 times to research the company (try queries like:
   "[company] company culture glassdoor", "[company] tech stack engineering blog",
   "[company] funding news 2024").
3. Once you have enough context, write the full report. Do not call any more tools after
   you start writing the report.

TOOL (when calling a tool output ONLY the tool call — nothing else in that message):
<tool>web_search</tool><input>your search query</input>

REPORT FORMAT — use these exact section headers with emoji:

## 🏢 Company Snapshot
What the company does, estimated size and stage (startup / scale-up / enterprise),
industry position, and culture signals from your research.

## 🎯 Role Breakdown
**Must-Have Skills:**
- (bullet list)

**Nice-to-Have Skills:**
- (bullet list)

**Seniority Level:** Junior / Mid / Senior / Staff / unclear — one-line rationale.

## 🚩 Red Flags
Honest assessment of: vague scope, contradictory requirements, "rockstar/ninja/10x"
language, excessive skill-stacking, missing salary info, on-call expectations, or signs
of churn and poor process. Write "None identified." only if genuinely clean.

## 🤝 Fit Assessment
3–5 sentences on what kind of candidate would genuinely thrive here. Be direct — if the
role looks like a grind or a stepping stone, say so.

## 💬 Interview Talking Points
- (3–5 bullets — specific angles to emphasise based on the JD requirements)

## ❓ Questions to Ask the Interviewer
1. [Specific question tied to something in the JD or your research]
2. [Question about team structure, process, or day-to-day reality]
3. [Question about success metrics, growth, or biggest current challenge]

RULES:
- Always run searches before writing — never skip the research phase.
- Reference specific details from the JD, not generic boilerplate.
- Red flags must be honest and concrete, not vague or softened.
- Interview questions must be tailored to THIS role, not copy-pasted generics.
"""

# ── Tools ─────────────────────────────────────────────────────────────────────
def tool_web_search(query: str) -> str:
    """DuckDuckGo search with Wikipedia fallback."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=4):
                snippet = r.get("body", "").strip()
                title   = r.get("title", "")
                url     = r.get("href", "")
                results.append(f"**{title}**\n{snippet}\nSource: {url}")
        if results:
            return "\n\n---\n\n".join(results)
        print("  [ddg] no results, falling back to Wikipedia")
    except Exception as e:
        print(f"  [ddg error] {e} — falling back to Wikipedia")

    # Wikipedia fallback
    try:
        r = SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 3, "format": "json"},
            timeout=15,
        )
        results = r.json().get("query", {}).get("search", [])
        if not results:
            return "No results found."
        title = results[0]["title"]
        r2 = SESSION.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "titles": title, "prop": "extracts",
                    "exintro": False, "explaintext": True, "redirects": 1, "format": "json"},
            timeout=15,
        )
        for pid, pg in r2.json().get("query", {}).get("pages", {}).items():
            if pid != "-1" and pg.get("extract"):
                return f"Wikipedia: {pg['title']}\n\n{pg['extract'][:2000]}"
        return "\n".join(
            f"- {s['title']}: {re.sub('<.*?>', '', s.get('snippet', ''))}"
            for s in results
        )
    except Exception as e:
        return f"Search error: {e}"


TOOLS = {"web_search": tool_web_search}

# ── LLM wrapper ───────────────────────────────────────────────────────────────
_last_call = 0.0

def call_llm(msgs: list, max_tokens: int = 800) -> str:
    global _last_call
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set.")
    gap = time.time() - _last_call
    if gap < 3.5:
        time.sleep(3.5 - gap)
    r = None
    for attempt in range(3):
        _last_call = time.time()
        r = requests.post(
            LLM_API_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": msgs,
                "max_tokens": max_tokens,
                "temperature": 0.3,
            },
            timeout=120,
        )
        if r.status_code == 429:
            wait = 15 * (attempt + 1)
            print(f"  [429] rate limited — waiting {wait}s")
            time.sleep(wait)
            continue
        break
    if r is None or not r.ok:
        status = r.status_code if r is not None else "no response"
        body   = r.text[:200] if r is not None else ""
        print(f"  [LLM error] {status}: {body}")
        if r is not None:
            r.raise_for_status()
        raise RuntimeError("LLM call failed with no response")
    return r.json()["choices"][0]["message"]["content"].strip()


# ── Tool-call parser ──────────────────────────────────────────────────────────
def parse_tool_call(text: str):
    m = re.search(r"<tool>(.*?)</tool>\s*<input>(.*?)</input>", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, None


# ── Agent loop ────────────────────────────────────────────────────────────────
MAX_TOOL_TURNS = 5   # searches before we force the final report
MAX_TOTAL_TURNS = 8

def analyse_job(job_description: str) -> str:
    if not job_description.strip():
        return "_Please paste a job description above and click **Analyse**._"
    if not GROQ_API_KEY:
        return (
            "**Configuration error:** `GROQ_API_KEY` is not set.\n\n"
            "Add it to a `.env` file in the project root:\n"
            "```\nGROQ_API_KEY=your_key_here\n```\n"
            "Get a free key at [console.groq.com](https://console.groq.com)."
        )

    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Analyse this job description:\n\n{job_description.strip()}"},
    ]

    tool_turns = 0

    for turn in range(MAX_TOTAL_TURNS):
        # After enough searches, give the model a larger budget for the report
        force_report = tool_turns >= MAX_TOOL_TURNS
        max_tok = 2800 if force_report or turn >= 4 else 700

        try:
            resp = call_llm(msgs, max_tokens=max_tok)
        except Exception as e:
            return f"**LLM error:** {e}"

        print(f"  [turn {turn + 1}] {resp[:200]}")

        tool, inp = parse_tool_call(resp)

        if tool and tool in TOOLS and not force_report:
            result = TOOLS[tool](inp)
            print(f"  [{tool}({inp[:60]!r})] → {result[:150]}")
            tool_turns += 1
            msgs += [
                {"role": "assistant", "content": resp},
                {
                    "role": "user",
                    "content": (
                        f"<result>\n{result}\n</result>\n\n"
                        "Continue your research or, if you have enough context, write the full report."
                    ),
                },
            ]
        else:
            # No tool call (or forced) — this is the final report
            return resp

    # Loop exhausted without a clean report — ask for it explicitly
    msgs.append({
        "role": "user",
        "content": "You have gathered enough information. Write the complete analysis report now.",
    })
    try:
        return call_llm(msgs, max_tokens=2800)
    except Exception as e:
        return f"**Error generating report:** {e}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
PLACEHOLDER = (
    "Paste the full job description here...\n\n"
    "Tip: include the company name, role title, responsibilities, and requirements "
    "for the best analysis."
)

with gr.Blocks(title="JobLens", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🔍 JobLens
**AI-powered job description analyser**

Paste any job posting and get a structured breakdown: company research, role analysis,
red flags, fit assessment, and tailored interview prep — powered by Groq + LLaMA 3.3.

---
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            jd_input = gr.Textbox(
                label="Job Description",
                placeholder=PLACEHOLDER,
                lines=22,
                max_lines=50,
            )
            analyse_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")
            gr.Markdown(
                "_Analysis takes ~30–60 seconds while the agent researches the company._",
                elem_classes=["hint"],
            )

        with gr.Column(scale=1):
            report_output = gr.Markdown(
                value="_Your report will appear here._",
                label="Analysis Report",
            )

    analyse_btn.click(
        fn=analyse_job,
        inputs=jd_input,
        outputs=report_output,
        show_progress=True,
    )

    gr.Markdown(
        """
---
*Built with [Gradio](https://gradio.app) · [Groq](https://groq.com) ·
llama-3.3-70b-versatile ·
[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)*
"""
    )

if __name__ == "__main__":
    print(f"GROQ_API_KEY: {'set ✓' if GROQ_API_KEY else 'NOT SET ✗'}")
    demo.launch(debug=True, share=False)
