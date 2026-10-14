from app.services.bm25 import BM25Index


def test_bm25_ranks_exact_evidence_first():
    index = BM25Index(
        [
            "The annual leave policy grants twenty paid days.",
            "Database backups run every night.",
            "The office cafeteria opens at eight.",
        ]
    )

    hits = index.search("annual leave policy", limit=3)

    assert hits[0].item_index == 0
    assert hits[0].score > 0
