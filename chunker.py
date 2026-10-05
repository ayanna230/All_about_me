"""
Step-by-step chunker for the Ayanna RAG project.

The strategy (structure-aware chunking):
1. Walk the markdown file line by line, tracking which headers (## and ###)
   are currently "active" — this becomes each chunk's breadcrumb.
2. Every time we hit a new header, we close off whatever content we were
   accumulating and turn it into a chunk (if it had real content).
3. Special case: the FAQ section (## 8. FAQ) has no ### subheadings — it's
   one big block of Q/A pairs. We detect that block by its breadcrumb and
   split it further, one chunk per Q/A pair, since each pair is already a
   perfect self-contained retrieval unit.
"""

import re
import json
from pathlib import Path

SOURCE_FILE = Path("ayanna_jack_source_document.md")
OUTPUT_FILE = Path("chunks.json")

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def split_into_raw_sections(text: str):
    """
    Walk the document and group lines under their active heading path.
    Returns a list of dicts: {"breadcrumb": [...], "text": "..."}
    """
    lines = text.splitlines()

    sections = []
    heading_stack = []          # e.g. ["3. Skills", "AI & Machine Learning"]
    current_content = []

    def flush():
        content = "\n".join(current_content).strip()
        if content:  # skip empty sections (e.g. a header with no body yet)
            sections.append({
                "breadcrumb": list(heading_stack),
                "text": content,
            })

    for line in lines:
        match = HEADER_RE.match(line)
        if match:
            # New heading found -> close off the previous section first
            flush()
            current_content = []

            level = len(match.group(1))   # 1 for '#', 2 for '##', 3 for '###'
            title = match.group(2).strip()

            # Adjust the heading stack to reflect the new level.
            # level 1 -> stack = [title]
            # level 2 -> stack = [title]      (we treat H1 as just the doc title, not a breadcrumb layer)
            # level 3 -> stack = [h2, title]
            if level == 1:
                heading_stack = []  # doc title, not useful as breadcrumb
            elif level == 2:
                heading_stack = [title]
            elif level == 3:
                heading_stack = heading_stack[:1] + [title]
        else:
            current_content.append(line)

    flush()  # don't forget the last section in the file
    return sections


def split_faq_block(faq_text: str):
    """
    Split a block of '**Q: ...**\\nA: ...' pairs into individual Q/A chunks.
    """
    # Each question starts with a line like "**Q: ...**"
    parts = re.split(r"(?=^\*\*Q:)", faq_text, flags=re.MULTILINE)
    qa_chunks = []
    for part in parts:
        part = part.strip()
        if part:
            qa_chunks.append(part)
    return qa_chunks


def build_chunks():
    text = SOURCE_FILE.read_text(encoding="utf-8")
    raw_sections = split_into_raw_sections(text)

    chunks = []
    for section in raw_sections:
        breadcrumb = section["breadcrumb"]
        body = section["text"]

        if not breadcrumb:
            # Content outside any heading (e.g. the italic "compiled from..."
            # note under the H1 title) is meta-commentary about the document,
            # not a fact about Ayanna -- skip it as a retrievable chunk.
            continue

        is_faq = len(breadcrumb) == 1 and breadcrumb[0].strip().endswith("FAQ")

        if is_faq:
            for qa in split_faq_block(body):
                chunks.append({
                    "breadcrumb": breadcrumb + ["Q&A"],
                    "text": qa,
                    "chunk_type": "faq_pair",
                })
        else:
            chunks.append({
                "breadcrumb": breadcrumb,
                "text": body,
                "chunk_type": "section",
            })

    # Add final fields: an id, a "header context" prefix, and char length
    final_chunks = []
    for i, c in enumerate(chunks):
        breadcrumb_str = " > ".join(c["breadcrumb"]) if c["breadcrumb"] else "(intro)"
        full_text = f"[{breadcrumb_str}]\n{c['text']}"
        final_chunks.append({
            "id": i,
            "breadcrumb": breadcrumb_str,
            "chunk_type": c["chunk_type"],
            "text": full_text,
            "char_count": len(full_text),
        })

    return final_chunks


if __name__ == "__main__":
    chunks = build_chunks()

    OUTPUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Total chunks: {len(chunks)}")
    lengths = [c["char_count"] for c in chunks]
    print(f"Char length -> min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}")
    print(f"\nSaved to {OUTPUT_FILE}\n")
    print("--- Preview of first 3 chunks ---")
    for c in chunks[:3]:
        print(f"\n#{c['id']} [{c['chunk_type']}] ({c['char_count']} chars)")
        print(c["text"])
