from pathlib import Path

from app.ingestion import build_chunks
from app.utils import clean_text, tokenize

DATA = Path(__file__).resolve().parents[1] / "data"


def chunks_for(name):
    return build_chunks(DATA / name, 350, 40)


def find(chunks, **meta):
    return [c for c in chunks if all(c.metadata[k] == v for k, v in meta.items())]


def test_rupee_symbol_and_bullets_are_repaired():
    text = "\n".join(c.text for c in chunks_for("sample_5.pdf"))
    assert "\u20b925,000" in text
    assert "\u25a0" not in text and "(cid:" not in text


def test_sections_carry_real_section_numbers_and_pages():
    chunks = chunks_for("sample_5.pdf")
    (foreclosure,) = find(chunks, section_id="6.2")
    assert "3% of the outstanding principal" in foreclosure.text
    assert foreclosure.metadata["section"].startswith("Section 6.2: Foreclosure")
    assert foreclosure.metadata["page_start"] == 3
    assert foreclosure.text.startswith("[FinBase Personal Loans Master Policy")


def test_subsection_spanning_pages_reports_both_pages():
    (age,) = find(chunks_for("sample_5.pdf"), section_id="2.1")
    assert age.metadata["pages"] == "1-2"
    assert "Minimum Age: 21 years" in age.text


def test_tables_become_rows_with_headers():
    (slabs,) = find(chunks_for("sample_5.pdf"), section_id="3")
    assert "| Loan Slab Code | Minimum Amount |" in slabs.text
    assert "| PL-PRIME | \u20b95,00,001 | \u20b910,00,000 |" in slabs.text


def test_faq_items_are_individual_chunks():
    chunks = chunks_for("sample_5.pdf")
    (q1,) = find(chunks, faq_id="Q001")
    assert "foreclosure charge" in q1.text.lower()
    assert len([c for c in chunks if c.metadata["faq_id"]]) == 100


def test_table_of_contents_is_dropped_and_document_info_kept():
    chunks = chunks_for("sample_5.pdf")
    assert not any("(#section-" in c.text for c in chunks)
    (info,) = find(chunks, section_id="0")
    assert "Document Code: FB-POL-PL-2026-V4" in info.text


def test_chunks_respect_size_budget_and_have_unique_ids():
    for name in ("sample_1.pdf", "sample_3.pdf", "sample_6.pdf"):
        chunks = chunks_for(name)
        assert len({c.chunk_id for c in chunks}) == len(chunks)
        assert max(len(c.text.split()) for c in chunks) <= 350 + 40


def test_clean_text_and_tokenize():
    assert clean_text("a \u00a0 b\r\nPage 3 of 9\n\n\n\nc") == "a b\n\nc"
    tokens = tokenize("Code MET-PL-1301 costs \u20b91,500")
    assert "met-pl-1301" in tokens and "1301" in tokens
