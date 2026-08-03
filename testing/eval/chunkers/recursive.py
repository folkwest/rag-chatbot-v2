"""Recursive structure-aware chunking strategy.

Splits text along a hierarchy of separators (headers, paragraphs, newlines,
sentences) and preserves atomic units (code blocks, HTML tables) from being
split when they fit within the maximum chunk size.
"""

import re
from typing import Optional

from testing.eval.chunkers.base import BaseChunker
from testing.eval.models import Chunk, PositionMetadata, RawUnit, count_tokens


# Patterns for atomic units
_CODE_FENCE_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
_HTML_TABLE_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


class RecursiveChunker(BaseChunker):
    """Recursively splits text along structural boundaries.

    Respects a separator hierarchy (headers > paragraphs > newlines > sentences)
    and preserves atomic units (code blocks, HTML tables) intact.
    """

    # Separator hierarchy from highest to lowest priority
    SEPARATORS = [
        re.compile(r"(?m)(?=^#{1,6}\s)"),  # Level 0: Markdown headers
        "\n\n",                             # Level 1: Paragraph boundaries
        "\n",                               # Level 2: Single newlines
        ". ",                               # Level 3a: Sentence period
        "? ",                               # Level 3b: Sentence question
        "! ",                               # Level 3c: Sentence exclamation
    ]

    def __init__(self, max_size: int = 500) -> None:
        """Initialize with maximum chunk size in whitespace-delimited tokens.

        Args:
            max_size: Maximum number of tokens per chunk (default 500).
        """
        self.max_size = max_size

    @property
    def strategy_name(self) -> str:
        """Return the name identifying this chunking strategy."""
        return "recursive"

    def chunk(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Accept raw units and produce positioned chunks.

        Algorithm:
        1. Build combined text with character-to-raw_unit offset mapping
        2. Detect atomic units (code blocks, HTML tables)
        3. Recursively split using separator hierarchy
        4. Merge adjacent small pieces up to max_size
        5. Map produced text segments back to source raw units for position metadata
        """
        if not raw_units:
            return []

        # Build combined text and character offset mapping
        combined_text, char_to_unit_idx = self._build_combined_text(raw_units)

        if not combined_text.strip():
            return []

        # Detect atomic unit spans (character ranges that shouldn't be split)
        atomic_spans = self._detect_atomic_spans(combined_text)

        # Recursively split — produces (start, end) ranges into combined_text
        piece_ranges = self._recursive_split(combined_text, atomic_spans, 0, 0, len(combined_text))

        # Merge adjacent small pieces up to max_size tokens
        piece_ranges = self._merge_small_pieces(combined_text, piece_ranges)

        # Build chunks with position metadata
        doc_id = raw_units[0].doc_id
        chunks: list[Chunk] = []

        for start, end in piece_ranges:
            piece_text = combined_text[start:end]
            if not piece_text.strip():
                continue

            position_metadata = self._compute_position_metadata(
                start, end - 1, char_to_unit_idx, raw_units
            )

            chunk_id = f"recursive_{doc_id}_{len(chunks)}"
            token_count = count_tokens(piece_text)

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=piece_text,
                    doc_id=doc_id,
                    strategy="recursive",
                    position_metadata_range=position_metadata,
                    token_count=token_count,
                )
            )

        return chunks

    def _merge_small_pieces(
        self, text: str, piece_ranges: list[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Merge adjacent small pieces until they approach max_size tokens.

        After recursive splitting, many pieces may be well below max_size.
        This pass greedily merges consecutive pieces into larger chunks,
        stopping when adding the next piece would exceed max_size.
        """
        if not piece_ranges:
            return []

        merged: list[tuple[int, int]] = []
        current_start, current_end = piece_ranges[0]
        current_tokens = count_tokens(text[current_start:current_end])

        for i in range(1, len(piece_ranges)):
            next_start, next_end = piece_ranges[i]
            next_tokens = count_tokens(text[next_start:next_end])

            # Can we merge? Check if combined text fits within max_size
            # The merged range spans from current_start to next_end
            combined_tokens = count_tokens(text[current_start:next_end])

            if combined_tokens <= self.max_size:
                # Merge: extend current range to include this piece
                current_end = next_end
                current_tokens = combined_tokens
            else:
                # Can't merge: flush current and start new
                merged.append((current_start, current_end))
                current_start = next_start
                current_end = next_end
                current_tokens = next_tokens

        # Flush the last accumulated range
        merged.append((current_start, current_end))
        return merged

    def _build_combined_text(
        self, raw_units: list[RawUnit]
    ) -> tuple[str, list[tuple[int, int, int]]]:
        """Build combined text and a mapping from character ranges to raw unit indices.

        Returns:
            Tuple of (combined_text, list of (char_start, char_end, unit_index) tuples)
        """
        parts: list[str] = []
        char_to_unit: list[tuple[int, int, int]] = []
        offset = 0

        for idx, unit in enumerate(raw_units):
            text = unit.text
            start = offset
            parts.append(text)
            offset += len(text)
            char_to_unit.append((start, offset - 1, idx))

            # Add separator between units
            if idx < len(raw_units) - 1:
                parts.append("\n\n")
                offset += 2

        combined = "".join(parts)
        return combined, char_to_unit

    def _detect_atomic_spans(self, text: str) -> list[tuple[int, int]]:
        """Detect character ranges of atomic units (code blocks, HTML tables).

        Returns:
            Sorted list of (start, end) character positions for atomic units.
            End is exclusive.
        """
        spans: list[tuple[int, int]] = []

        for match in _CODE_FENCE_PATTERN.finditer(text):
            spans.append((match.start(), match.end()))

        for match in _HTML_TABLE_PATTERN.finditer(text):
            spans.append((match.start(), match.end()))

        spans.sort(key=lambda s: s[0])
        return spans

    def _is_atomic(self, start: int, end: int, atomic_spans: list[tuple[int, int]]) -> bool:
        """Check if a text range [start, end) is entirely contained within an atomic span."""
        for a_start, a_end in atomic_spans:
            if start >= a_start and end <= a_end:
                return True
        return False

    def _split_position_inside_atomic(self, pos: int, atomic_spans: list[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """If pos is inside an atomic span, return that span. Otherwise None."""
        for a_start, a_end in atomic_spans:
            if a_start < pos < a_end:
                return (a_start, a_end)
        return None

    def _recursive_split(
        self,
        text: str,
        atomic_spans: list[tuple[int, int]],
        separator_level: int,
        abs_start: int,
        abs_end: int,
    ) -> list[tuple[int, int]]:
        """Recursively split text using the separator hierarchy.

        Works with absolute positions in the combined text.

        Args:
            text: The full combined text.
            atomic_spans: List of (start, end) ranges for atomic units.
            separator_level: Current level in the separator hierarchy.
            abs_start: Start position in combined text for this segment.
            abs_end: End position (exclusive) in combined text for this segment.

        Returns:
            List of (start, end) ranges representing the final pieces.
        """
        segment = text[abs_start:abs_end]

        # Base case: segment fits within max_size
        if count_tokens(segment) <= self.max_size:
            return [(abs_start, abs_end)]

        # Check if this segment is entirely an atomic unit
        if self._is_atomic(abs_start, abs_end, atomic_spans):
            # Atomic unit — keep as-is even if oversized
            return [(abs_start, abs_end)]

        # If all separators exhausted, split at token boundary
        if separator_level >= len(self.SEPARATORS):
            return self._split_at_token_boundary(text, abs_start, abs_end)

        # Try splitting with current separator
        split_positions = self._find_split_positions(
            text, self.SEPARATORS[separator_level], abs_start, abs_end, atomic_spans
        )

        # If no valid split positions found, try next separator level
        if not split_positions:
            return self._recursive_split(text, atomic_spans, separator_level + 1, abs_start, abs_end)

        # Build sub-segments from split positions
        sub_ranges = self._build_sub_ranges(abs_start, abs_end, split_positions)

        # Process each sub-segment recursively
        result: list[tuple[int, int]] = []
        for seg_start, seg_end in sub_ranges:
            seg_text = text[seg_start:seg_end]
            if not seg_text.strip():
                continue

            if count_tokens(seg_text) <= self.max_size:
                result.append((seg_start, seg_end))
            elif self._is_atomic(seg_start, seg_end, atomic_spans):
                result.append((seg_start, seg_end))
            else:
                # Recurse with next separator level
                sub_result = self._recursive_split(
                    text, atomic_spans, separator_level + 1, seg_start, seg_end
                )
                result.extend(sub_result)

        return result

    def _find_split_positions(
        self,
        text: str,
        separator,
        abs_start: int,
        abs_end: int,
        atomic_spans: list[tuple[int, int]],
    ) -> list[int]:
        """Find valid split positions within a segment, respecting atomic boundaries.

        Returns a sorted list of absolute positions where splits should occur.
        A split at position p means segment is divided into [...p) and [p...).
        """
        segment = text[abs_start:abs_end]
        positions: list[int] = []

        if isinstance(separator, re.Pattern):
            for match in separator.finditer(segment):
                abs_pos = abs_start + match.start()
                # Don't split inside atomic spans
                if not self._split_position_inside_atomic(abs_pos, atomic_spans):
                    positions.append(abs_pos)
        else:
            # String separator — find all occurrences
            search_start = 0
            while True:
                idx = segment.find(separator, search_start)
                if idx == -1:
                    break
                abs_pos = abs_start + idx
                # Don't split inside atomic spans
                if not self._split_position_inside_atomic(abs_pos, atomic_spans):
                    positions.append(abs_pos)
                search_start = idx + 1

        # Filter out splits at the very start (would create empty first piece)
        positions = [p for p in positions if p > abs_start]

        return positions

    def _build_sub_ranges(
        self, abs_start: int, abs_end: int, split_positions: list[int]
    ) -> list[tuple[int, int]]:
        """Build sub-ranges from split positions.

        Given split positions, creates non-overlapping (start, end) ranges.
        """
        ranges: list[tuple[int, int]] = []
        prev = abs_start

        for pos in split_positions:
            if pos > prev:
                ranges.append((prev, pos))
            prev = pos

        # Add the final segment
        if prev < abs_end:
            ranges.append((prev, abs_end))

        return ranges

    def _split_at_token_boundary(
        self, text: str, abs_start: int, abs_end: int
    ) -> list[tuple[int, int]]:
        """Split text at token boundaries when all separators are exhausted.

        Splits the segment into pieces of at most max_size tokens each,
        returning absolute position ranges.
        """
        segment = text[abs_start:abs_end]
        tokens = segment.split()
        pieces: list[tuple[int, int]] = []

        token_idx = 0
        while token_idx < len(tokens):
            batch = tokens[token_idx : token_idx + self.max_size]
            piece_text = " ".join(batch)

            # Find the position of this piece in the segment
            if token_idx == 0:
                # First piece starts at the beginning (possibly after leading whitespace)
                piece_start_in_seg = segment.find(batch[0])
            else:
                # Subsequent pieces: find after the previous piece ended
                prev_end = pieces[-1][1] - abs_start
                piece_start_in_seg = segment.find(batch[0], prev_end)

            if piece_start_in_seg == -1:
                piece_start_in_seg = 0

            # End position: after the last token in this batch
            last_token_start = segment.find(batch[-1], piece_start_in_seg)
            piece_end_in_seg = last_token_start + len(batch[-1])

            pieces.append((abs_start + piece_start_in_seg, abs_start + piece_end_in_seg))
            token_idx += self.max_size

        return pieces

    def _compute_position_metadata(
        self,
        piece_start: int,
        piece_end: int,
        char_to_unit: list[tuple[int, int, int]],
        raw_units: list[RawUnit],
    ) -> PositionMetadata:
        """Compute position metadata range from character offsets.

        Finds the first and last raw units that contribute to the given
        character range and merges their position metadata.
        """
        first_unit_idx: Optional[int] = None
        last_unit_idx: Optional[int] = None

        for char_start, char_end, unit_idx in char_to_unit:
            # Check if this raw unit's character range overlaps with the piece
            if char_start <= piece_end and char_end >= piece_start:
                if first_unit_idx is None:
                    first_unit_idx = unit_idx
                last_unit_idx = unit_idx

        if first_unit_idx is None or last_unit_idx is None:
            # Fallback: use first raw unit
            first_unit_idx = 0
            last_unit_idx = 0

        first_meta = raw_units[first_unit_idx].position_metadata
        last_meta = raw_units[last_unit_idx].position_metadata

        return PositionMetadata(
            line_number_start=first_meta.line_number_start,
            line_number_end=last_meta.line_number_end or last_meta.line_number_start,
            page_number_start=first_meta.page_number_start,
            page_number_end=last_meta.page_number_end or last_meta.page_number_start,
            item_id=first_meta.item_id,
        )
