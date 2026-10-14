from app.evaluation import evaluate_dataset


def test_evaluation_calculates_dense_and_hybrid_recall():
    class ExactEmbedder:
        dimension = 2

        def embed_documents(self, texts):
            return [[1.0, 0.0] if "leave" in text else [0.0, 1.0] for text in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    dataset = {
        "corpus": [
            {"id": "leave", "text": "annual leave policy"},
            {"id": "backup", "text": "nightly database backup"},
        ],
        "queries": [{"query": "vacation allowance", "relevant_chunk_ids": ["leave"]}],
    }

    report = evaluate_dataset(dataset, ExactEmbedder(), k=1)

    assert report.dense_recall == 1.0
    assert report.hybrid_recall == 1.0
