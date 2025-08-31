#!/usr/bin/env python3
import argparse, html, json, os, re, sys
from pathlib import Path
from datetime import datetime

# ---------- Helpers ----------
FNAME_TS = re.compile(r".*?(\d{4}-\d{2}-\d{2}[_T-]\d{2}[:\-]\d{2}[:\-]\d{2})")

def guess_session_time(p: Path, first_msg_time: str|None) -> datetime:
    # Prefer time from JSON if present; else parse filename; else mtime.
    for ts in [first_msg_time]:
        if ts:
            try:
                # Try various common iso-ish formats
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S"):
                    try: return datetime.strptime(ts[:19], fmt)
                    except: pass
            except: pass
    m = FNAME_TS.match(p.name)
    if m:
        raw = m.group(1).replace("_","T").replace("-","-").replace(" ", "T")
        raw = raw.replace("-", ":", 2) if raw.count(":")<2 else raw
        try: return datetime.fromisoformat(raw[:19])
        except: pass
    return datetime.fromtimestamp(p.stat().st_mtime)

def read_messages(jsonl_path: Path):
    msgs = []
    with jsonl_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # Only take "message" records; tolerate schema drift
            if (obj.get("type") or obj.get("event")) == "message" or "role" in obj:
                role = obj.get("role") or obj.get("source") or "assistant"
                content = obj.get("content") or obj.get("text") or ""
                msgs.append({"role": role, "content": str(content)})
    return msgs

def to_md(msgs):
    out = []
    for m in msgs:
        role = m["role"]
        content = m["content"].rstrip()
        out.append(f"**{role}:**\n\n{content}\n")
    return "\n".join(out)

def esc(s): return html.escape(s, quote=False)

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser(description="Index Codex CLI sessions into HTML + Markdown.")
    ap.add_argument("--input-dir", default=str(Path.home()/".codex"/"sessions"))
    ap.add_argument("--out-dir", default="codex_sessions_index")
    ap.add_argument("--preview-chars", type=int, default=160, help="Chars to show from first user msg")
    args = ap.parse_args()

    in_dir = Path(args.input_dir).expanduser()
    out_dir = Path(args.out_dir)
    sess_dir = out_dir / "sessions"
    out_dir.mkdir(parents=True, exist_ok=True)
    sess_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(2)

    rows = []
    mega_sections = []
    for p in sorted(in_dir.glob("*.jsonl")):
        msgs = read_messages(p)
        if not msgs:
            continue
        first_user = next((m for m in msgs if m["role"] == "user"), msgs[0])
        first_user_text = first_user["content"].replace("\n", " ").strip()
        first_time = None  # extend here if your schema includes timestamps
        when = guess_session_time(p, first_time)
        num_user = sum(1 for m in msgs if m["role"] == "user")
        num_asst = sum(1 for m in msgs if m["role"] in ("assistant","system"))
        basename = p.name
        md_path = sess_dir / (basename.replace(".jsonl",".md"))
        md_path.write_text(f"# {basename}\n\n_Date_: {when.isoformat()}\n\n{to_md(msgs)}\n", encoding="utf-8")

        rows.append({
            "fname": basename,
            "when": when,
            "count": len(msgs),
            "user": num_user,
            "asst": num_asst,
            "preview": first_user_text[:args.preview_chars],
            "link": f"sessions/{md_path.name}",
        })
        mega_sections.append(f"## {basename} — {when.isoformat()}\n\n{to_md(msgs)}\n")

    # Sort newest first
    rows.sort(key=lambda r: r["when"], reverse=True)

    # Write mega markdown
    (out_dir/"mega_transcript.md").write_text("# Codex Sessions — Mega Transcript\n\n" + "\n\n---\n\n".join(mega_sections), encoding="utf-8")

    # Write HTML index
    html_rows = []
    for r in rows:
        html_rows.append(
            f"<tr>"
            f"<td>{esc(r['when'].strftime('%Y-%m-%d %H:%M:%S'))}</td>"
            f"<td><a href='{esc(r['link'])}'>{esc(r['fname'])}</a></td>"
            f"<td style='text-align:right'>{r['count']}</td>"
            f"<td style='text-align:right'>{r['user']}</td>"
            f"<td style='text-align:right'>{r['asst']}</td>"
            f"<td>{esc(r['preview'])}</td>"
            f"</tr>"
        )
    index_html = f"""<!doctype html>
<html><head>
<meta charset="utf-8"/>
<title>Codex Sessions Index</title>
<style>
body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 8px; vertical-align: top; }}
th {{ position: sticky; top: 0; background: #fafafa; }}
code {{ background:#f3f3f3; padding:2px 4px; border-radius:4px; }}
.controls {{ margin-bottom: 12px; }}
</style>
<script>
function filterTable() {{
  const q = document.getElementById('q').value.toLowerCase();
  for (const tr of document.querySelectorAll('#t tbody tr')) {{
    tr.style.display = tr.innerText.toLowerCase().includes(q) ? '' : 'none';
  }}
}}
</script>
</head><body>
<h1>Codex Sessions Index</h1>
<div class="controls">
  <input id="q" oninput="filterTable()" placeholder="Filter…"
         style="width: 40%; padding: 6px 8px; font-size: 14px"/>
  <span style="margin-left: 12px"><a href="mega_transcript.md">mega_transcript.md</a></span>
</div>
<table id="t">
  <thead><tr>
    <th>Date</th><th>Session</th><th>#msgs</th><th>#user</th><th>#asst</th><th>First prompt (preview)</th>
  </tr></thead>
  <tbody>
    {''.join(html_rows)}
  </tbody>
</table>
</body></html>
"""
    (out_dir/"index.html").write_text(index_html, encoding="utf-8")
    print(f"Wrote: {out_dir/'index.html'}")
    print(f"Wrote: {out_dir/'mega_transcript.md'}")
    print(f"Wrote: {sess_dir}/*")

if __name__ == "__main__":
    main()
