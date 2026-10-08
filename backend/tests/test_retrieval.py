from app.retrieval import RerankerUnavailable, reciprocal_rank_fusion
from conftest import make_settings


def ids(result, key="section_id"):
    return [h.metadata[key] for h in result.hits]


def test_rrf_prefers_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "a", "d"]])
    assert [i for i, _ in fused][:2] == ["a", "c"]


def test_dense_retrieves_foreclosure_section(retriever):
    result = retriever.retrieve("foreclosure charge when closing a personal loan before 24 months", mode="dense")
    sections = {(h.metadata["file_name"], h.metadata["section_id"]) for h in result.hits}
    assert ("sample_5.pdf", "6.2") in sections or ("sample_5.pdf", "23") in sections


def test_sparse_stage_ranks_exact_identifier_first_and_hybrid_pool_contains_it(retriever):
    query = "What is the remediation time for MET-PL-1301?"
    best = retriever._sparse(query, 5)[0]
    assert "MET-PL-1301" in retriever._texts[best]
    pool = retriever.retrieve(query, mode="hybrid", top_k=retriever._settings.rerank_pool)
    assert any("MET-PL-1301" in h.text for h in pool.hits)


def test_hits_expose_dense_similarity_and_top_similarity(retriever):
    result = retriever.retrieve("UPI daily limit", mode="hybrid")
    assert 0.0 < result.top_similarity <= 1.0
    assert all(-1.0 <= h.dense_similarity <= 1.0 for h in result.hits)


def test_unknown_mode_is_rejected(retriever):
    try:
        retriever.retrieve("x", mode="bogus")
    except ValueError as exc:
        assert "bogus" in str(exc)
    else:
        raise AssertionError("expected ValueError")


class ReverseReranker:
    def score(self, query, passages):
        return [float(i) for i in range(len(passages))]


class BrokenReranker:
    def score(self, query, passages):
        raise RerankerUnavailable("no model")


def test_reranker_reorders_pool(built_index, embedder):
    from app.retrieval import HybridRetriever

    settings, collection = built_index
    base = HybridRetriever(collection, embedder, settings).retrieve("foreclosure charge", mode="hybrid")
    reranked = HybridRetriever(collection, embedder, settings, reranker=ReverseReranker()).retrieve(
        "foreclosure charge", mode="hybrid_rerank"
    )
    assert reranked.mode == "hybrid_rerank"
    assert reranked.hits[0].chunk_id != base.hits[0].chunk_id


def test_unavailable_reranker_falls_back_to_hybrid(built_index, embedder):
    from app.retrieval import HybridRetriever

    settings, collection = built_index
    result = HybridRetriever(collection, embedder, settings, reranker=BrokenReranker()).retrieve(
        "foreclosure charge", mode="hybrid_rerank"
    )
    assert result.mode == "hybrid" and result.notes


def test_catalog_lists_all_six_documents(retriever):
    docs = retriever.documents()
    assert len(docs) == 6 and all(d["code"] for d in docs)


def test_index_is_reused_when_unchanged(built_index, embedder):
    from app.index_store import load_or_build_collection

    settings, collection = built_index
    before = collection.count()
    again = load_or_build_collection(settings, embedder)
    assert again.count() == before
