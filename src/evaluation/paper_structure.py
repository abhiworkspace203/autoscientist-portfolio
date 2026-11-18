"""
Paper structure analysis (Member 3, Week 4).

Reads the text of a paper (PDF text, LaTeX or Markdown) and reports which
standard academic sections it has, how many figures, tables and references it
contains, and how long the abstract is. Used for both the human holdout papers
and the agent's generated paper, so both are measured the same way.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

SECTION_ALIASES: Dict[str, List[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction"],
    "related_work": ["related work", "background", "literature review", "prior work"],
    "method": ["method", "methods", "methodology", "materials and methods", "approach",
               "model", "proposed method", "theory", "theoretical framework"],
    "experiments": ["experiments", "experiment", "experimental setup", "experimental results",
                    "evaluation", "simulation", "simulations", "numerical results"],
    "results": ["results", "results and discussion", "findings"],
    "discussion": ["discussion", "analysis", "limitations"],
    "conclusion": ["conclusion", "conclusions", "summary", "conclusion and future work",
                   "conclusions and outlook", "outlook"],
    "references": ["references", "bibliography"],
}

REQUIRED_SECTIONS = ["abstract", "introduction", "method", "results", "conclusion", "references"]

_NAME_TO_KEY = {name: key for key, names in SECTION_ALIASES.items() for name in names}
_HEADING = re.compile(
    r"^\s*(?:\\section\*?\{|\\subsection\*?\{|#{1,4}\s*)?"
    r"(?:(?:\d+(?:\.\d+)*|[IVX]+|[A-Z])\.?\s+)?"
    r"([A-Za-z][A-Za-z &\-]{2,40}?)\s*\}?\s*[:.]?\s*$"
)
_ABSTRACT_ENV = re.compile(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.S)
_FIGURE = re.compile(r"(?:^|\n)\s*(?:Figure|Fig\.)\s*([A-Z]?\d+)\s*[:.|]", re.I)
_TABLE = re.compile(r"(?:^|\n)\s*Table\s*([A-Z]?\d+)\s*[:.|]", re.I)
_LATEX_FIG = re.compile(r"\\begin\{figure\*?\}")
_LATEX_TAB = re.compile(r"\\begin\{table\*?\}")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_NUMBERED_REF = re.compile(r"^\s*\[(\d{1,3})\]", re.M)
_YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
_BIBITEM = re.compile(r"\\bibitem|@(?:article|inproceedings|misc|book)\s*\{", re.I)


def find_sections(text: str) -> List[str]:
    """Canonical section names in the order they first appear."""
    found: List[str] = []
    if _ABSTRACT_ENV.search(text):
        found.append("abstract")
    for line in text.splitlines():
        if len(line) > 60:
            continue
        m = _HEADING.match(line)
        if not m:
            continue
        key = _NAME_TO_KEY.get(m.group(1).strip().lower())
        if key and key not in found:
            found.append(key)
    return found


def _section_text(text: str, start_names: List[str]) -> str:
    """Text after the first heading named in start_names, up to the next heading."""
    lines = text.splitlines()
    out: List[str] = []
    inside = False
    for line in lines:
        m = _HEADING.match(line) if len(line) <= 60 else None
        name = m.group(1).strip().lower() if m else ""
        if inside and name in _NAME_TO_KEY:
            break
        if inside:
            out.append(line)
        elif name in start_names:
            inside = True
    return "\n".join(out)


def abstract_text(text: str) -> str:
    m = _ABSTRACT_ENV.search(text)
    if m:
        return m.group(1).strip()
    return _section_text(text, SECTION_ALIASES["abstract"]).strip()


def count_references(text: str) -> int:
    """Number of bibliography entries (numbered [n], \\bibitem / BibTeX, or year-bearing lines)."""
    bib = len(_BIBITEM.findall(text))
    if bib:
        return bib
    refs = _section_text(text, SECTION_ALIASES["references"])
    if not refs:
        return 0
    numbered = {int(n) for n in _NUMBERED_REF.findall(refs)}
    if numbered:
        return len(numbered)
    return sum(1 for line in refs.splitlines() if _YEAR.search(line) and len(line) > 30)


def analyse_structure(text: str) -> Dict[str, Any]:
    """Structure summary of a paper's text."""
    sections = find_sections(text)
    figures = {m.lower() for m in _FIGURE.findall(text)}
    tables = {m.lower() for m in _TABLE.findall(text)}
    abstract = abstract_text(text)
    return {
        "sections": sections,
        "missing_sections": [s for s in REQUIRED_SECTIONS if s not in sections],
        "word_count": len(text.split()),
        "abstract_words": len(abstract.split()),
        "figures": max(len(figures), len(_LATEX_FIG.findall(text)), len(_MD_IMAGE.findall(text))),
        "tables": max(len(tables), len(_LATEX_TAB.findall(text))),
        "references": count_references(text),
    }


def pdf_text(pdf_path: str) -> str:
    """Plain text of a PDF (empty string if it cannot be read)."""
    try:
        import fitz
    except ImportError:
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            if doc.needs_pass:
                return ""
            return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""
