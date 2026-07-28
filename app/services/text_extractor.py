import io
from pypdf import PdfReader
from docx import Document


class UnsupportedFileTypeError(Exception):
    def __init__(self, filename: str):
        self.filename = filename
        super().__init__(f"Unsupported file type:'{filename}'. Supported types: .txt, .pdf, .docx. Legacy .doc is not supported."
            f"Please save as .docx first."
        )


class TextExtractor:
    '''Turning an uploaded file's raw bytes into plain text'''
    def extract(self, filename: str, file_bytes: bytes) -> str:
        lower_filename = filename.lower()

        if lower_filename.endswith('.txt'):
            return self._extract_from_txt(file_bytes)
        elif lower_filename.endswith('.pdf'):
            return self._extract_from_pdf(file_bytes)
        elif lower_filename.endswith(".docx"):
            return self._extract_from_docx(file_bytes)
        else:
            raise UnsupportedFileTypeError(filename)


    def _extract_from_txt(self, file_bytes) -> str:
        """Plain text files — just decode the raw bytes."""
        return file_bytes.decode("utf-8")


    def _extract_from_pdf(self, file_bytes) -> str:
        """
        Extracts text from every page of a PDF and joins it into one string. 
        Only works for text-based PDFs — scanned/image-only PDFs would need OCR (out of scope here) and come back empty.
        """
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages_text)


    def _extract_from_docx(self, file_bytes: bytes) -> str:
        """
        Extracts text from a .docx file's paragraphs, joined with
        newlines. Only handles body paragraphs — tables, headers/footers,
        and embedded objects are not extracted (a reasonable scope
        limit for a first version).
        """
        document = Document(io.BytesIO(file_bytes))
        paragraphs_text = [p.text for p in document.paragraphs]
        return "\n".join(paragraphs_text)


text_extractor = TextExtractor()