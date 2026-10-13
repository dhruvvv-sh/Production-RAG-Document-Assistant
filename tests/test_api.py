from app.services.pdf import PdfPage


def test_liveness(client):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_and_get_document(client, monkeypatch):
    monkeypatch.setattr(
        "app.services.ingestion.extract_pages",
        lambda _: [PdfPage(number=1, text="Evidence sentence. " * 50)],
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("sample.pdf", b"%PDF-test-content", "application/pdf")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["page_count"] == 1
    assert payload["chunk_count"] > 0

    fetched = client.get(f"/api/v1/documents/{payload['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == payload["id"]

    duplicate = client.post(
        "/api/v1/documents",
        files={"file": ("renamed.pdf", b"%PDF-test-content", "application/pdf")},
    )
    assert duplicate.status_code == 200
    assert duplicate.json()["id"] == payload["id"]


def test_rejects_non_pdf(client):
    response = client.post(
        "/api/v1/documents",
        files={"file": ("notes.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 415
