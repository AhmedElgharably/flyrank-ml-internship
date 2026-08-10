#!/usr/bin/env python3
"""
Capstone Scout — Research Agent for FlyRank ML Internship
Builds on FL-06 spec. Searches arXiv, filters duplicates, downloads PDFs,
synthesizes notes with Claude, saves to research-notes/, generates digest.

Usage:
    python capstone_scout.py

Requirements:
    pip install requests feedparser pdfplumber anthropic
    Set ANTHROPIC_API_KEY environment variable
"""

import os
import sys
import re
import json
import hashlib
import requests
import feedparser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber

# ─── CONFIG ──────────────────────────────────────────────────────────
NOTES_DIR = Path.home() / "flyrank-ml-internship" / "research-notes"
PAPERS_DIR = Path.home() / "flyrank-ml-internship" / "papers"
MAX_PAPERS_PER_RUN = 10
MAX_PDF_SIZE_MB = 50
DAYS_BACK = 7

SEARCH_QUERIES = [
    "content refresh SEO machine learning",
    "learning to rank ranking model",
    "CTR prediction click-through rate",
    "content embedding document embedding semantic search",
]

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("ERROR: Set ANTHROPIC_API_KEY environment variable")
    sys.exit(1)

# ─── SETUP ───────────────────────────────────────────────────────────
NOTES_DIR.mkdir(parents=True, exist_ok=True)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

# ─── ARXIV SEARCH ────────────────────────────────────────────────────
def search_arxiv(query: str, days_back: int = DAYS_BACK) -> List[Dict]:
    """Search arXiv for papers published in last N days."""
    base_url = "http://export.arxiv.org/api/query"
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": 20,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️  arXiv search failed for '{query}': {e}")
        return []

    feed = feedparser.parse(resp.text)
    papers = []

    for entry in feed.entries:
        published = datetime(*entry.published_parsed[:6])
        if published < datetime.now() - timedelta(days=days_back):
            continue

        papers.append({
            "title": entry.title.replace("\n", " ").strip(),
            "authors": [a.name for a in entry.authors],
            "published": published.strftime("%Y-%m-%d"),
            "summary": entry.summary.replace("\n", " ").strip(),
            "pdf_url": entry.link.replace("/abs/", "/pdf/") + ".pdf",
            "arxiv_id": entry.id.split("/abs/")[-1],
            "source": "arXiv",
        })

    return papers

# ─── DUPLICATE DETECTION ─────────────────────────────────────────────
def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def is_duplicate(title: str) -> bool:
    """Check if a paper with similar title already exists in notes."""
    normalized = normalize_title(title)

    if not NOTES_DIR.exists():
        return False

    for note_file in NOTES_DIR.glob("*.md"):
        with open(note_file, "r", encoding="utf-8") as f:
            content = f.read()
            # Extract title from first line (# Title)
            match = re.search(r"^# (.+)$", content, re.MULTILINE)
            if match:
                existing_title = normalize_title(match.group(1))
                # Exact match or high similarity
                if existing_title == normalized:
                    return True
                # Simple word overlap similarity
                existing_words = set(existing_title.split())
                new_words = set(normalized.split())
                if len(existing_words) > 0 and len(new_words) > 0:
                    overlap = len(existing_words & new_words) / max(len(existing_words), len(new_words))
                    if overlap > 0.85:
                        return True

    return False

# ─── PDF DOWNLOAD ────────────────────────────────────────────────────
def download_pdf(pdf_url: str, arxiv_id: str) -> Optional[Path]:
    """Download PDF to papers dir. Returns path or None."""
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = PAPERS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}-{safe_id}.pdf"

    try:
        resp = requests.get(pdf_url, timeout=60, stream=True)
        resp.raise_for_status()

        # Check size
        content_length = resp.headers.get("content-length")
        if content_length and int(content_length) > MAX_PDF_SIZE_MB * 1024 * 1024:
            print(f"⚠️  PDF too large (> {MAX_PDF_SIZE_MB}MB): {arxiv_id}")
            return None

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Verify size after download
        if pdf_path.stat().st_size > MAX_PDF_SIZE_MB * 1024 * 1024:
            pdf_path.unlink()
            print(f"⚠️  PDF too large after download: {arxiv_id}")
            return None

        return pdf_path

    except requests.RequestException as e:
        print(f"⚠️  PDF download failed for {arxiv_id}: {e}")
        return None

# ─── PDF TEXT EXTRACTION ─────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path, max_pages: int = 5) -> str:
    """Extract text from first N pages of PDF."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages[:max_pages]):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text.strip()
    except Exception as e:
        print(f"⚠️  PDF extraction failed for {pdf_path.name}: {e}")
        return ""

# ─── CLAUDE SYNTHESIS ────────────────────────────────────────────────
def synthesize_with_claude(paper: Dict, pdf_text: str) -> Optional[str]:
    """Send paper to Claude for structured synthesis."""

    system_prompt = """You are Capstone Scout, a research assistant for an ML Engineering intern working on content refresh scoring at FlyRank.

Format the paper into a markdown research note with this EXACT structure:

# [Paper Title]
**Authors:** [names]
**Date:** [date]
**Source:** [arXiv / Semantic Scholar]

## One-Line Problem
[Core problem in one sentence]

## Data & Scale
[Dataset and size, or "Not specified" if not in abstract/first pages]

## Method
[Main architecture or approach from abstract/first pages]

## Key Numbers
[Best metric, baseline comparison, improvement — from abstract/first pages only. If not present, write "Not found in preview pages."]

## Honest Caveat
[Biggest limitation mentioned or obvious from the approach]

## FlyRank Takeaway
[One practical connection to content refresh scoring. If none, say honestly: "No direct connection, but the method could be adapted."]

---
*Scouted by Capstone Scout · [date]*

RULES:
- Direct, warm, evidence-first, no buzzwords.
- If a number seems suspicious, flag it with ⚠️.
- Never claim the paper "proves" anything causal unless stated.
- Only use information from the provided text. Do not hallucinate.
- If the text is too short or unclear, say so honestly."""

    user_prompt = f"""Paper title: {paper['title']}
Authors: {', '.join(paper['authors'])}
Date: {paper['published']}
Source: {paper['source']}
Abstract: {paper['summary']}

First pages text:
{pdf_text[:4000]}

Please synthesize this into the structured markdown format."""

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 2000,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["content"][0]["text"]

    except Exception as e:
        print(f"⚠️  Claude synthesis failed for {paper['title']}: {e}")
        return None

# ─── SAVE NOTE ───────────────────────────────────────────────────────
def save_note(paper: Dict, content: str) -> Path:
    """Save synthesized note to research-notes/."""
    safe_title = re.sub(r"[^\w\s-]", "", paper['title'])[:50].strip().replace(" ", "-")
    filename = f"{paper['published']}-{safe_title}.md"
    note_path = NOTES_DIR / filename

    with open(note_path, "w", encoding="utf-8") as f:
        f.write(content)

    return note_path

# ─── DIGEST GENERATION ───────────────────────────────────────────────
def generate_digest(processed: List[Dict], skipped: int, failed: int) -> str:
    """Generate Monday morning digest."""

    lines = [
        f"# Capstone Scout Digest — {datetime.now().strftime('%Y-%m-%d')}",
        "",
        f"**This week:** {len(processed)} new papers processed. {skipped} skipped (already read or off-topic). {failed} failed (download or synthesis error).",
        "",
        "## Top picks",
        "",
    ]

    for i, p in enumerate(processed[:3], 1):
        lines.append(f"{i}. **{p['title']}** — {p.get('relevance', 'Relevance TBD')}")

    if len(processed) > 3:
        lines.append(f"\n...and {len(processed) - 3} more. See `research-notes/` for full summaries.")

    lines.extend([
        "",
        "## Action items",
        "",
        "- [ ] Review the top pick — does it connect to your capstone?",
        "- [ ] Spot-check 1 in 3 notes against the original PDF",
        "- [ ] Update search keywords if your capstone direction has shifted",
        "",
        "---",
        "*These are starting points, not replacements. Open the PDF for the full method and limitations.*",
    ])

    return "\n".join(lines)

# ─── MAIN LOOP ───────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("CAPSTONE SCOUT — Starting run")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    all_papers = []

    # Step 1: Search
    print("\n🔍 Searching arXiv...")
    for query in SEARCH_QUERIES:
        papers = search_arxiv(query)
        print(f"   '{query}': {len(papers)} papers found")
        all_papers.extend(papers)

    # Deduplicate by arxiv_id
    seen_ids = set()
    unique_papers = []
    for p in all_papers:
        if p["arxiv_id"] not in seen_ids:
            seen_ids.add(p["arxiv_id"])
            unique_papers.append(p)

    print(f"\n📊 Total unique papers from arXiv: {len(unique_papers)}")

    # Step 2: Filter duplicates
    new_papers = []
    skipped = 0
    for p in unique_papers:
        if is_duplicate(p["title"]):
            skipped += 1
            print(f"   ⏭️  Skipped (already read): {p['title'][:60]}...")
        else:
            new_papers.append(p)

    print(f"\n📁 New papers to process: {len(new_papers)} (skipped {skipped})")

    # Limit to MAX_PAPERS_PER_RUN
    if len(new_papers) > MAX_PAPERS_PER_RUN:
        print(f"   ⚠️  Limiting to {MAX_PAPERS_PER_RUN} papers (config limit)")
        new_papers = new_papers[:MAX_PAPERS_PER_RUN]

    # Step 3: Download, extract, synthesize, save
    processed = []
    failed = 0

    for i, paper in enumerate(new_papers, 1):
        print(f"\n📝 [{i}/{len(new_papers)}] {paper['title'][:70]}...")

        # Download PDF
        pdf_path = download_pdf(paper["pdf_url"], paper["arxiv_id"])
        if not pdf_path:
            failed += 1
            continue
        print(f"   ✅ PDF downloaded: {pdf_path.name}")

        # Extract text
        pdf_text = extract_text_from_pdf(pdf_path)
        if not pdf_text:
            print(f"   ⚠️  Could not extract text, using abstract only")
            pdf_text = paper["summary"]
        else:
            print(f"   ✅ Extracted {len(pdf_text)} chars from PDF")

        # Synthesize with Claude
        print(f"   🤖 Sending to Claude for synthesis...")
        note_content = synthesize_with_claude(paper, pdf_text)
        if not note_content:
            failed += 1
            continue

        # Save note
        note_path = save_note(paper, note_content)
        print(f"   ✅ Note saved: {note_path.name}")

        processed.append(paper)

    # Step 4: Generate digest
    print("\n" + "=" * 60)
    print("GENERATING DIGEST...")
    print("=" * 60)

    digest = generate_digest(processed, skipped, failed)
    digest_path = NOTES_DIR / f"digest-{datetime.now().strftime('%Y-%m-%d')}.md"
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(digest)

    print(f"\n📬 Digest saved: {digest_path.name}")
    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print(f"Papers found:      {len(unique_papers)}")
    print(f"Already read:      {skipped}")
    print(f"Processed:         {len(processed)}")
    print(f"Failed:            {failed}")
    print(f"Notes saved to:    {NOTES_DIR}")
    print(f"Digest saved to:   {digest_path}")
    print("=" * 60)

    # Print digest
    print("\n--- DIGEST PREVIEW ---\n")
    print(digest)

if __name__ == "__main__":
    main()
