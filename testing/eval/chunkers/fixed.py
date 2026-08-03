"""Fixed-size chunking strategy.

Splits concatenated text into chunks of a fixed token count with
configurable overlap, ignoring document structure entirely.
"""

from testing.eval.chunkers.base import BaseChunker
from testing.eval.models import Chunk, PositionMetadata, RawUnit, count_tokens


class FixedSizeChunker(BaseChunker):
    """Chunker that splits text into fixed-size token windows with overlap.

    Concatenates all Raw_Unit text, splits by whitespace tokens, and slides
    a window of `chunk_size` tokens with stride = chunk_size - overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        """Initialize the fixed-size chunker.

        Args:
            chunk_size: Number of whitespace-delimited tokens per chunk.
                        Must be between 50 and 10000 inclusive.
            overlap: Number of tokens shared between consecutive chunks.
                     Must be >= 0 and < chunk_size.

        Raises:
            ValueError: If overlap >= chunk_size, or parameters are out of range.
        """
        if chunk_size < 50 or chunk_size > 10000:
            raise ValueError(
                f"chunk_size must be between 50 and 10000, got {chunk_size}"
            )
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            )
        self._chunk_size = chunk_size
        self._overlap = overlap

    @property
    def strategy_name(self) -> str:
        """Return the name identifying this chunking strategy."""
        return "fixed_size"

    def chunk(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Split raw units into fixed-size token chunks.

        1. Concatenates all Raw_Unit text with space separators
        2. Splits by whitespace into a flat token list
        3. Tracks which token indices map to which Raw_Unit
        4. Slides a window of chunk_size tokens with stride = chunk_size - overlap
        5. For each window, determines contributing Raw_Units for position metadata

        Args:
            raw_units: List of RawUnit objects to chunk.

        Returns:
            List of Chunk objects. Empty list if input contains only whitespace.
        """
        if not raw_units:
            return []

        # Build flat token list and track which RawUnit each token belongs to
        tokens: list[str] = []
        token_to_unit_index: list[int] = []

        for unit_idx, raw_unit in enumerate(raw_units):
            unit_tokens = raw_unit.text.split()
            for token in unit_tokens:
                tokens.append(token)
                token_to_unit_index.append(unit_idx)

        # If no tokens (whitespace-only input), return empty
        if not tokens:
            return []

        # Determine doc_id from first raw unit
        doc_id = raw_units[0].doc_id

        # Slide window
        stride = self._chunk_size - self._overlap
        chunks: list[Chunk] = []
        index = 0
        start = 0

        while start < len(tokens):
            end = min(start + self._chunk_size, len(tokens))
            window_tokens = tokens[start:end]

            # Determine contributing Raw_Units for this window
            unit_indices = set(token_to_unit_index[start:end])
            first_unit_idx = min(unit_indices)
            last_unit_idx = max(unit_indices)

            # Build position_metadata_range from first and last contributing units
            first_meta = raw_units[first_unit_idx].position_metadata
            last_meta = raw_units[last_unit_idx].position_metadata

            position_range = PositionMetadata(
                line_number_start=first_meta.line_number_start,
                line_number_end=last_meta.line_number_end
                if last_meta.line_number_end is not None
                else last_meta.line_number_start,
                page_number_start=first_meta.page_number_start,
                page_number_end=last_meta.page_number_end
                if last_meta.page_number_end is not None
                else last_meta.page_number_start,
                item_id=first_meta.item_id
                if first_meta.item_id == last_meta.item_id
                else (
                    f"{first_meta.item_id}-{last_meta.item_id}"
                    if first_meta.item_id and last_meta.item_id
                    else first_meta.item_id or last_meta.item_id
                ),
            )

            chunk_text = " ".join(window_tokens)
            chunk_id = f"fixed_size_{doc_id}_{index}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=position_range,
                    token_count=len(window_tokens),
                )
            )

            index += 1
            start += stride

            # Avoid infinite loop if stride is somehow 0 (shouldn't happen due to validation)
            if stride <= 0:
                break

        return chunks
