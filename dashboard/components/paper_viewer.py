import base64
import re
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
GENERATED_HTML = OUTPUT_DIR / "generated_research_paper.html"
GENERATED_TEX = OUTPUT_DIR / "generated_research_paper.tex"
GENERATED_PDF = OUTPUT_DIR / "generated_research_paper.pdf"


def _tex_to_pdf_bytes(tex_path: Path) -> bytes:
    """Generate a real PDF from .tex source using reportlab (no LaTeX needed)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    import io

    tex_content = tex_path.read_text(encoding="utf-8", errors="ignore")

    # Extract key sections from LaTeX
    def extract(pattern, text, default=""):
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else default

    title = extract(r"\\title\{(.+?)\}", tex_content, "Autonomous Scientific Discovery — Research Paper")
    abstract = extract(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", tex_content, "")
    # Clean latex commands
    def clean(s):
        s = re.sub(r"\\[a-zA-Z]+\*?\{([^}]*)\}", r"\1", s)
        s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
        s = re.sub(r"[{}]", "", s)
        return s.strip()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2.5*cm, rightMargin=2.5*cm,
                            topMargin=2.5*cm, bottomMargin=2.5*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18,
                                 spaceAfter=12, alignment=TA_CENTER)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=11,
                                leading=16, spaceAfter=8, alignment=TA_JUSTIFY)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"],
                                   fontSize=13, spaceBefore=16, spaceAfter=6)

    story = [
        Paragraph(clean(title), title_style),
        Spacer(1, 0.4*cm),
    ]

    if abstract:
        story.append(Paragraph("Abstract", section_style))
        story.append(Paragraph(clean(abstract), body_style))
        story.append(Spacer(1, 0.3*cm))

    # Extract sections
    sections = re.findall(r"\\section\{(.+?)\}(.*?)(?=\\section|\Z)", tex_content, re.DOTALL)
    for sec_title, sec_body in sections:
        story.append(Paragraph(clean(sec_title), section_style))
        paragraphs = re.split(r"\n{2,}", clean(sec_body))
        for para in paragraphs:
            para = para.strip()
            if para:
                story.append(Paragraph(para, body_style))

    if not sections:
        story.append(Paragraph("Full LaTeX Source", section_style))
        story.append(Preformatted(tex_content[:3000], styles["Code"]))

    doc.build(story)
    return buf.getvalue()


def render_paper_viewer() -> None:
    """Preview and download the generated research paper."""

    # --- Show HTML version (always works) ---
    if GENERATED_HTML.exists():
        st.caption(f"Showing: {GENERATED_HTML}")
        html_content = GENERATED_HTML.read_text(encoding="utf-8")
        st.components.v1.html(html_content, height=800, scrolling=True)

        # --- Real PDF download via reportlab ---
        if GENERATED_TEX.exists():
            if st.button("⬇️ Generate & Download PDF"):
                with st.spinner("Generating PDF..."):
                    try:
                        pdf_bytes = _tex_to_pdf_bytes(GENERATED_TEX)
                        st.download_button(
                            "📄 Download PDF Now",
                            data=pdf_bytes,
                            file_name="generated_research_paper.pdf",
                            mime="application/pdf",
                        )
                        st.success("✅ Real PDF generated successfully!")
                    except Exception as e:
                        st.error(f"PDF generation failed: {e}")
        else:
            st.info("No .tex source found. Run main.py --synthesize-latex first.")

    else:
        st.info("No generated paper found yet. Run the pipeline first:")
        st.code("python main.py --topic 'Your Topic' --enable-critic --synthesize-latex")

        st.markdown("**Or upload a PDF to preview:**")
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_file is not None:
            pdf_bytes = uploaded_file.getvalue()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" width="100%" '
                f'height="750px" style="border:none;border-radius:8px;"></iframe>',
                unsafe_allow_html=True,
            )
