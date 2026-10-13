from dataclasses import dataclass
from io import BytesIO

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class InvalidPdfError(ValueError):
    pass


@dataclass(frozen=True)
class PdfPage:
    number: int
    text: str


def extract_pages(content: bytes) -> list[PdfPage]:
    try:
        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise InvalidPdfError("Encrypted PDFs are not supported")
        pages = [
            PdfPage(number=index, text=(page.extract_text() or "").strip())
            for index, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, EOFError, ValueError) as exc:
        if isinstance(exc, InvalidPdfError):
            raise
        raise InvalidPdfError("The uploaded file is not a readable PDF") from exc

    if not pages:
        raise InvalidPdfError("The PDF has no pages")
    if not any(page.text for page in pages):
        raise InvalidPdfError("The PDF contains no extractable text")
    return pages
