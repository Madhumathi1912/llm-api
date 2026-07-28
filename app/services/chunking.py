class Chunker:
    """
    Owns ONE responsibility: splitting text into overlapping chunks.
    Configurable via constructor (chunk_size, overlap) so RagService
    doesn't need to know or care about the splitting strategy's details
    — it just calls chunker.chunk(text) and gets a list back.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        if chunk_size <= overlap:
            raise ValueError("chunk size must be greater than overlap")
        self._chunk_size = chunk_size
        self._overlap = overlap


    def chunk(self, text: str) -> list[str]:
        """
        Splits text into roughly self._chunk_size-character chunks, with
        self._overlap characters repeated between consecutive chunks so
        a sentence/idea split across a boundary isn't lost entirely.
 
        This is a simple character-based splitter for learning purposes.
        Production systems often chunk by token count (not characters)
        and/or respect paragraph/sentence boundaries rather than cutting
        mid-word — a natural upgrade to make to this class later without
        touching any of its callers.
        """
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self._chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            start += self._chunk_size - self._overlap
        return chunks
    

chunker = Chunker()