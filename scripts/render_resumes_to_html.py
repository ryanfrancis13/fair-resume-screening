"""Generate visual, printable HTML document previews for resumes in the dataset.

Opens directly in your browser with A4 document styling, and can be saved
as a real PDF with Ctrl+P (Print -> Save as PDF).
"""
import html
import json
import re
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("data/previews")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {{
    --primary: #1e293b;
    --accent: #2563eb;
    --text: #334155;
    --muted: #64748b;
    --border: #e2e8f0;
    --bg: #f8fafc;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0f172a;
    color: var(--text);
    padding: 30px 15px;
    line-height: 1.5;
  }}

  .nav-bar {{
    max-width: 900px;
    margin: 0 auto 20px auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #1e293b;
    padding: 12px 24px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .nav-bar h2 {{ color: #f8fafc; font-size: 16px; font-weight: 600; }}
  .btn-print {{
    background: #3b82f6;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }}
  .btn-print:hover {{ background: #2563eb; }}

  /* A4 Paper Sheet Simulation */
  .page {{
    max-width: 850px;
    min-height: 1100px;
    background: #ffffff;
    margin: 0 auto 30px auto;
    padding: 50px 60px;
    border-radius: 8px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    position: relative;
  }}

  .badge-tag {{
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 15px;
  }}
  .badge-good {{ background: #dcfce7; color: #15803d; }}
  .badge-potential {{ background: #fef9c3; color: #a16207; }}
  .badge-nofit {{ background: #fee2e2; color: #b91c1c; }}

  .candidate-header {{
    border-bottom: 2px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 24px;
  }}
  .candidate-name {{
    font-size: 26px;
    font-weight: 700;
    color: var(--primary);
    letter-spacing: -0.5px;
  }}
  .candidate-title {{
    font-size: 15px;
    color: var(--accent);
    font-weight: 600;
    margin-top: 4px;
  }}
  .candidate-contact {{
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-top: 10px;
    font-size: 12px;
    color: var(--muted);
  }}

  .section {{
    margin-bottom: 22px;
  }}
  .section-title {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--primary);
    border-bottom: 1.5px solid var(--border);
    padding-bottom: 5px;
    margin-bottom: 12px;
  }}

  .section-content {{
    font-size: 13px;
    color: var(--text);
    white-space: pre-wrap;
    line-height: 1.6;
  }}

  .job-context-box {{
    max-width: 850px;
    margin: 0 auto 30px auto;
    background: #1e293b;
    border-radius: 8px;
    padding: 24px 30px;
    color: #cbd5e1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  }}
  .job-context-box h3 {{
    color: #f8fafc;
    font-size: 15px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  .job-desc {{
    font-size: 12.5px;
    line-height: 1.6;
    color: #94a3b8;
    max-height: 250px;
    overflow-y: auto;
    white-space: pre-wrap;
    background: #0f172a;
    padding: 15px;
    border-radius: 6px;
    border: 1px solid #334155;
  }}

  @media print {{
    body {{ background: white; padding: 0; }}
    .nav-bar, .job-context-box {{ display: none; }}
    .page {{
      box-shadow: none;
      padding: 0;
      margin: 0;
      max-width: 100%;
    }}
  }}
</style>
</head>
<body>

<div class="nav-bar">
  <h2>Resume Document Preview (A4 Format)</h2>
  <button class="btn-print" onclick="window.print()">Print / Save as PDF</button>
</div>

<div class="job-context-box">
  <h3>
    Applied Role & Fit Evaluation
    <span class="badge-tag {badge_class}">{label}</span>
  </h3>
  <div class="job-desc">{job_description}</div>
</div>

<div class="page">
  <div class="candidate-header">
    <div class="candidate-name">{name}</div>
    <div class="candidate-title">{headline}</div>
    <div class="candidate-contact">
      <span>Location: {location}</span>
      <span>•</span>
      <span>Email: {email}</span>
      <span>•</span>
      <span>Source: Hugging Face Fit Corpus</span>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Resume Content & Experience</div>
    <div class="section-content">{body}</div>
  </div>
</div>

</body>
</html>
"""

def format_resume_html(row, idx: int) -> str:
    label = str(row["label"]).strip()
    badge_class = {
        "Good Fit": "badge-good",
        "Potential Fit": "badge-potential",
        "No Fit": "badge-nofit",
    }.get(label, "badge-potential")

    r_text = str(row["resume_text"]).strip()
    j_text = str(row["job_description_text"]).strip()

    # Extract first line of job description for headline
    j_lines = [line.strip() for line in j_text.splitlines() if line.strip()]
    headline = j_lines[0] if j_lines else "Candidate Profile"

    # Extract or infer name/location if available
    name = f"Candidate Profile #{idx:03d}"
    location = "San Francisco, CA / Remote"

    # Clean up common OCR spacing issues for display
    cleaned_body = r_text
    for section in ["Summary", "Skills", "Experience", "Education and Training", "Education"]:
        cleaned_body = re.sub(rf"(?<=[a-z])({section})", r"\n\n\1: ", cleaned_body)

    return HTML_TEMPLATE.format(
        title=f"Resume Preview #{idx:03d} - {label}",
        badge_class=badge_class,
        label=html.escape(label),
        job_description=html.escape(j_text),
        name=html.escape(name),
        headline=html.escape(headline[:80]),
        location=html.escape(location),
        email=f"applicant.{idx:03d}@example.com",
        body=html.escape(cleaned_body),
    )

def main():
    csv_path = Path("data/raw/hf_fit_train.csv")
    if not csv_path.exists():
        print("CSV not found!")
        return

    df = pd.read_csv(csv_path)

    # Export 1 Good Fit, 1 Potential Fit, and 1 No Fit sample as rendered HTML documents
    samples = [
        (df[df["label"] == "Good Fit"].iloc[0], 1, "good_fit_sample.html"),
        (df[df["label"] == "Potential Fit"].iloc[0], 2, "potential_fit_sample.html"),
        (df[df["label"] == "No Fit"].iloc[0], 3, "no_fit_sample.html"),
    ]

    for row, idx, filename in samples:
        out_file = OUTPUT_DIR / filename
        out_file.write_text(format_resume_html(row, idx), encoding="utf-8")
        print(f"Generated: {out_file.resolve()}")

    # Also render synthetic sample R01 for comparison
    r_synth_path = Path("data/sample/resumes/R01.json")
    j_synth_path = Path("data/sample/jobs/J01.json")
    if r_synth_path.exists() and j_synth_path.exists():
        r_synth = json.loads(r_synth_path.read_text(encoding="utf-8"))
        j_synth = json.loads(j_synth_path.read_text(encoding="utf-8"))
        synth_html = HTML_TEMPLATE.format(
            title="Synthetic R01 Preview - Senior Backend Engineer",
            badge_class="badge-good",
            label="Strong Fit (Reference Baseline)",
            job_description=html.escape(j_synth["text"]),
            name=html.escape(r_synth["baseline_slots"].get("NAME", "Jordan Avery")),
            headline=html.escape(j_synth.get("title", "Senior Backend Engineer")),
            location=html.escape(r_synth["baseline_slots"].get("LOCATION", "Boston, MA")),
            email=html.escape(r_synth["baseline_slots"].get("EMAIL", "jordan.avery@example.com")),
            body=html.escape(r_synth["text"]),
        )
        (OUTPUT_DIR / "synthetic_R01_sample.html").write_text(synth_html, encoding="utf-8")
        print(f"Generated: {(OUTPUT_DIR / 'synthetic_R01_sample.html').resolve()}")

    # Also generate a quick index viewer
    index_html = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Resume Previews Index</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: white; padding: 40px; text-align: center; }
  .card-container { display: flex; gap: 20px; justify-content: center; margin-top: 30px; flex-wrap: wrap; }
  .card { background: #1e293b; border-radius: 10px; padding: 25px; width: 280px; text-align: left; text-decoration: none; color: white; border: 1px solid #334155; transition: transform 0.2s; }
  .card:hover { transform: translateY(-5px); border-color: #3b82f6; }
  .card h3 { margin-bottom: 10px; }
  .card p { font-size: 13px; color: #94a3b8; }
</style>
</head>
<body>
  <h1>Resume Document Previews (A4 Format)</h1>
  <p style="color: #94a3b8; margin-top: 8px;">Click any sample below to view formatted as an A4 document. Press <b>Ctrl + P</b> to save as PDF.</p>
  <div class="card-container">
    <a class="card" href="synthetic_R01_sample.html" target="_blank">
      <h3 style="color: #60a5fa;">Synthetic Sample (R01)</h3>
      <p>Clean templated resume with all 9 cue slots visible and defined.</p>
    </a>
  </div>
</body>
</html>"""
    (OUTPUT_DIR / "index.html").write_text(index_html, encoding="utf-8")
    print(f"Index available at: {(OUTPUT_DIR / 'index.html').resolve()}")

if __name__ == "__main__":
    main()
