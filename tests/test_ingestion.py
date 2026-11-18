"""
Offline tests for the Member 3 ingestion pipelines.
No network needed: arXiv and NCBI calls are mocked, so these run in CI too.

Run:  pytest tests/test_ingestion.py -v
"""

import io
from datetime import date, datetime, timezone
from unittest.mock import patch

import arxiv
import pytest
from Bio import Entrez

from src.ingestion import arxiv_scraper, pubmed_scraper
from src.ingestion.schema import REQUIRED_FIELDS, Paper, deduplicate, load_papers, save_papers

# ------------------------------------------------------------------ fixtures
def make_arxiv_result(n: int) -> arxiv.Result:
    return arxiv.Result(
        entry_id=f"http://arxiv.org/abs/2410.0000{n}v1",
        published=datetime(2026, 10, n, tzinfo=timezone.utc),
        title=f"Paper  {n}\n  on agents",
        authors=[arxiv.Result.Author("Ada Lovelace"), arxiv.Result.Author("Alan Turing")],
        summary=f"Abstract number {n}\nwith newlines.",
        primary_category="cs.AI",
        categories=["cs.AI", "cs.LG"],
        links=[arxiv.Result.Link(f"http://arxiv.org/pdf/2410.0000{n}v1",
                                 title="pdf", content_type="application/pdf")],
    )


PUBMED_XML = b"""<?xml version="1.0" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2019//EN"
 "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_190101.dtd">
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation Status="MEDLINE" Owner="NLM">
   <PMID Version="1">39000001</PMID>
   <Article PubModel="Print">
    <Journal><JournalIssue CitedMedium="Internet"><PubDate><Year>2026</Year><Month>Sep</Month></PubDate></JournalIssue>
     <Title>Nature Methods</Title></Journal>
    <ArticleTitle>Deep learning for protein binding.</ArticleTitle>
    <Abstract>
     <AbstractText Label="BACKGROUND">Proteins bind.</AbstractText>
     <AbstractText Label="RESULTS">Models   predict it.</AbstractText>
    </Abstract>
    <AuthorList CompleteYN="Y">
     <Author ValidYN="Y"><LastName>Curie</LastName><ForeName>Marie</ForeName></Author>
     <Author ValidYN="Y"><CollectiveName>AI Consortium</CollectiveName></Author>
    </AuthorList>
    <ArticleDate DateType="Electronic"><Year>2026</Year><Month>09</Month><Day>15</Day></ArticleDate>
   </Article>
   <MeshHeadingList><MeshHeading><DescriptorName UI="D011506">Proteins</DescriptorName></MeshHeading></MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">39000001</ArticleId>
   <ArticleId IdType="doi">10.1000/xyz123</ArticleId>
   <ArticleId IdType="pmc">PMC1234567</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation Status="MEDLINE" Owner="NLM">
   <PMID Version="1">39000002</PMID>
   <Article PubModel="Print">
    <Journal><JournalIssue CitedMedium="Internet"><PubDate><Year>2026</Year></PubDate></JournalIssue><Title>J X</Title></Journal>
    <ArticleTitle>A paper with no abstract.</ArticleTitle>
   </Article>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">39000002</ArticleId></ArticleIdList></PubmedData>
 </PubmedArticle>
</PubmedArticleSet>
"""

ESEARCH_XML = b"""<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD esearch 20060628//EN"
 "https://eutils.ncbi.nlm.nih.gov/eutils/dtd/20060628/esearch.dtd">
<eSearchResult><Count>2</Count><RetMax>2</RetMax><RetStart>0</RetStart>
<IdList><Id>39000001</Id><Id>39000002</Id></IdList>
<TranslationSet/><QueryTranslation>x</QueryTranslation></eSearchResult>
"""


# ------------------------------------------------------------------ schema
def test_deduplicate_by_id_and_content():
    a = Paper("arxiv:1", "arxiv", "1", "T", "A", ["x"], "2026-01-01", "u")
    b = Paper("arxiv:1", "arxiv", "1", "T2", "A2", ["x"], "2026-01-01", "u")   # same id
    c = Paper("pubmed:9", "pubmed", "9", "t ", "a", ["y"], "2026-01-01", "u")  # same text
    assert len(deduplicate([a, b, c])) == 1


def test_save_and_load_roundtrip(tmp_path):
    p = Paper("arxiv:1", "arxiv", "1", "T", "A", ["x"], "2026-01-01", "u")
    path = save_papers([p], tmp_path / "out.json")
    assert load_papers(path)[0] == p


# ------------------------------------------------------------------ arXiv
def test_build_query_with_dates():
    q = arxiv_scraper.build_query(["cs.AI", "q-bio.BM"], date(2026, 10, 1), date(2026, 10, 11))
    assert q == "(cat:cs.AI OR cat:q-bio.BM) AND submittedDate:[202610010000 TO 202610112359]"


def test_build_query_requires_category():
    with pytest.raises(ValueError):
        arxiv_scraper.build_query([])


def test_arxiv_result_to_paper_fields():
    p = arxiv_scraper.result_to_paper(make_arxiv_result(3))
    assert p.paper_id == "arxiv:2410.00003v1"
    assert p.title == "Paper 3 on agents"            # whitespace cleaned
    assert p.pdf_url.endswith("2410.00003v1")
    assert p.published_date == "2026-10-03"
    assert all(getattr(p, f) for f in REQUIRED_FIELDS)


def test_arxiv_search_mocked():
    fake = [make_arxiv_result(i) for i in range(1, 6)]
    with patch.object(arxiv.Client, "results", return_value=iter(fake)):
        papers = arxiv_scraper.ArxivScraper().search(["cs.AI"], max_results=5)
    assert len(papers) == 5 and all(p.source == "arxiv" for p in papers)


def test_arxiv_network_timeout_returns_empty():
    with patch.object(arxiv.Client, "results", side_effect=TimeoutError("timed out")):
        assert arxiv_scraper.ArxivScraper().search(["cs.AI"]) == []


# ------------------------------------------------------------------ PubMed
def test_pubmed_record_parsing():
    records = Entrez.read(io.BytesIO(PUBMED_XML))
    p = pubmed_scraper.record_to_paper(records["PubmedArticle"][0])
    assert p.paper_id == "pubmed:39000001"
    assert p.abstract == "BACKGROUND: Proteins bind. RESULTS: Models predict it."
    assert p.authors == ["Marie Curie", "AI Consortium"]
    assert p.published_date == "2026-09-15"
    assert p.doi == "10.1000/xyz123"
    assert p.pdf_url and "PMC1234567" in p.pdf_url
    assert p.categories == ["Proteins"]


def test_pubmed_search_mocked_drops_no_abstract():
    with patch.object(Entrez, "esearch", return_value=io.BytesIO(ESEARCH_XML)), \
         patch.object(Entrez, "efetch", return_value=io.BytesIO(PUBMED_XML)):
        papers = pubmed_scraper.PubMedScraper(email="t@t.com").search("protein", max_results=5)
    assert [p.source_id for p in papers] == ["39000001"]       # 2nd has no abstract


def test_pubmed_network_timeout_returns_empty():
    s = pubmed_scraper.PubMedScraper(email="t@t.com", retries=2, backoff=0)
    with patch.object(Entrez, "esearch", side_effect=TimeoutError("timed out")):
        assert s.search("protein") == []
