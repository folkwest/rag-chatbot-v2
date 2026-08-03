"""Section-aware chunking strategy.

Splits text at document section boundaries (Markdown headers, SEC Item headings),
producing variable-size chunks that never cross section boundaries. Falls back to
paragraph-boundary splitting for documents without recognized headers.
"""

import re

from testing.eval.chunkers.base import BaseChunker
from testing.eval.models import Chunk, PositionMetadata, RawUnit, count_tokens


# Pattern matching Markdown header lines (one or more # followed by a space)
_MARKDOWN_HEADER_RE = re.compile(r"^(#+)\s", re.MULTILINE)


class SectionAwareChunker(BaseChunker):
    """Chunker that splits strictly at document section boundaries.

    Behavior depends on document type (detected from position_metadata):
    - Markdown (line_number): splits at header lines (^#+\\s)
    - SEC HTML (item_id): one chunk per Item section
    - PDF (page_number): one chunk per page
    - Plain text without headers: splits at paragraph boundaries (blank lines)
    """

    @property
    def strategy_name(self) -> str:
        """Return the name identifying this chunking strategy."""
        return "section_aware"

    def chunk(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Split raw units into section-aligned chunks.

        Args:
            raw_units: List of RawUnit objects to chunk.

        Returns:
            List of Chunk objects. Empty list if input is empty or whitespace-only.
        """
        if not raw_units:
            return []

        # Determine document type from first raw_unit's position_metadata
        first_meta = raw_units[0].position_metadata

        if first_meta.line_number_start is not None:
            return self._chunk_markdown(raw_units)
        elif first_meta.item_id is not None:
            return self._chunk_sec_html(raw_units)
        elif first_meta.page_number_start is not None:
            return self._chunk_pdf(raw_units)
        else:
            # Fallback: treat as plain text with paragraph splitting
            return self._chunk_paragraph_fallback(raw_units)

    def _chunk_markdown(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Chunk a Markdown/text document by splitting at header lines.

        Algorithm:
        1. Concatenate all raw_unit text with newlines
        2. Split at Markdown header lines (lines starting with # followed by space)
        3. Include any text before the first header as a separate chunk
        4. Discard empty sections (header immediately followed by another header)
        5. Each section gets position_metadata_range from contributing raw_units

        Falls back to paragraph-boundary splitting if no headers are found.
        """
        doc_id = raw_units[0].doc_id

        # Concatenate all text, preserving newlines between units.
        # Track character offset ranges for each raw_unit in the concatenated text.
        separator = "\n\n"
        parts: list[str] = []
        # Maps: (char_start, char_end) -> raw_unit index
        unit_char_ranges: list[tuple[int, int, int]] = []  # (start, end, unit_idx)
        offset = 0
        for idx, ru in enumerate(raw_units):
            if idx > 0:
                offset += len(separator)
            start = offset
            parts.append(ru.text)
            offset += len(ru.text)
            unit_char_ranges.append((start, offset, idx))

        full_text = separator.join(parts)

        # Check if there are any Markdown headers
        if not _MARKDOWN_HEADER_RE.search(full_text):
            # No headers found — fall back to paragraph splitting
            return self._chunk_paragraph_fallback(raw_units)

        # Split the text into sections at header boundaries.
        # Track character offset of each section in full_text.
        lines = full_text.split("\n")
        sections: list[tuple[str, int, int]] = []  # (section_text, char_start, char_end)
        current_section_lines: list[str] = []
        current_section_start: int = 0
        char_pos = 0

        for i, line in enumerate(lines):
            line_start = char_pos
            # Advance char_pos past this line (and the \n separator)
            char_pos += len(line) + (1 if i < len(lines) - 1 else 0)

            if _MARKDOWN_HEADER_RE.match(line):
                # Found a header — flush current section
                if current_section_lines:
                    section_text = "\n".join(current_section_lines)
                    sections.append(
                        (section_text, current_section_start, line_start)
                    )
                current_section_lines = [line]
                current_section_start = line_start
            else:
                current_section_lines.append(line)

        # Flush the last section
        if current_section_lines:
            section_text = "\n".join(current_section_lines)
            sections.append((section_text, current_section_start, len(full_text)))

        # Build chunks from non-empty sections
        chunks: list[Chunk] = []
        chunk_index = 0

        for section_text, sec_char_start, sec_char_end in sections:
            stripped = section_text.strip()
            if not stripped:
                continue

            # Discard empty sections: a section that contains only a header line
            # with no content below it (header immediately followed by another header)
            section_lines = stripped.split("\n")
            content_lines = [
                line for line in section_lines
                if line.strip() and not _MARKDOWN_HEADER_RE.match(line)
            ]
            if not content_lines and _MARKDOWN_HEADER_RE.match(section_lines[0]):
                # Section has a header but no content — discard
                continue

            # Determine position_metadata_range from character ranges
            position_range = self._position_from_char_range(
                sec_char_start, sec_char_end, unit_char_ranges, raw_units
            )

            chunk_id = f"section_aware_{doc_id}_{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=stripped,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=position_range,
                    token_count=count_tokens(stripped),
                )
            )
            chunk_index += 1

        return chunks

    def _position_from_char_range(
        self,
        sec_start: int,
        sec_end: int,
        unit_char_ranges: list[tuple[int, int, int]],
        raw_units: list[RawUnit],
    ) -> PositionMetadata:
        """Determine position_metadata_range from character offset range.

        Finds all raw_units whose character range overlaps with the section's
        character range in the concatenated text.
        """
        contributing_indices: list[int] = []
        for unit_start, unit_end, unit_idx in unit_char_ranges:
            # Check overlap between [sec_start, sec_end) and [unit_start, unit_end)
            if unit_end > sec_start and unit_start < sec_end:
                contributing_indices.append(unit_idx)

        if not contributing_indices:
            # Fallback to first unit
            contributing_indices = [0]

        first_unit = raw_units[contributing_indices[0]]
        last_unit = raw_units[contributing_indices[-1]]

        return self._merge_position_metadata(
            first_unit.position_metadata, last_unit.position_metadata
        )

    def _chunk_sec_html(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Chunk an SEC HTML document — one chunk per non-empty Raw_Unit.

        Each Raw_Unit already represents one Item section from ingestion.
        Preserves the item_id in position_metadata_range.
        """
        doc_id = raw_units[0].doc_id
        chunks: list[Chunk] = []
        chunk_index = 0

        for raw_unit in raw_units:
            text = raw_unit.text.strip()
            if not text:
                continue

            chunk_id = f"section_aware_{doc_id}_{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=PositionMetadata(
                        item_id=raw_unit.position_metadata.item_id,
                    ),
                    token_count=count_tokens(text),
                )
            )
            chunk_index += 1

        return chunks

    def _chunk_pdf(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Chunk a PDF document — each Raw_Unit (page) becomes one chunk."""
        doc_id = raw_units[0].doc_id
        chunks: list[Chunk] = []
        chunk_index = 0

        for raw_unit in raw_units:
            text = raw_unit.text.strip()
            if not text:
                continue

            chunk_id = f"section_aware_{doc_id}_{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=text,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=PositionMetadata(
                        page_number_start=raw_unit.position_metadata.page_number_start,
                        page_number_end=raw_unit.position_metadata.page_number_end,
                    ),
                    token_count=count_tokens(text),
                )
            )
            chunk_index += 1

        return chunks

    def _chunk_paragraph_fallback(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Fall back to paragraph-boundary splitting.

        For text documents without recognized headers:
        - Split at blank lines (one or more consecutive blank lines)
        - If no paragraph boundaries exist, produce a single chunk
        """
        doc_id = raw_units[0].doc_id

        # Concatenate all text with character range tracking
        separator = "\n\n"
        parts: list[str] = []
        unit_char_ranges: list[tuple[int, int, int]] = []
        offset = 0
        for idx, ru in enumerate(raw_units):
            if idx > 0:
                offset += len(separator)
            start = offset
            parts.append(ru.text)
            offset += len(ru.text)
            unit_char_ranges.append((start, offset, idx))

        full_text = separator.join(parts)

        # Split at paragraph boundaries (one or more blank lines)
        # Use re.split but track positions
        paragraph_pattern = re.compile(r"\n\s*\n")
        matches = list(paragraph_pattern.finditer(full_text))

        if not matches:
            # No paragraph boundaries — produce single chunk if non-empty
            stripped = full_text.strip()
            if not stripped:
                return []
            position_range = self._build_full_position_range(raw_units)
            return [
                Chunk(
                    chunk_id=f"section_aware_{doc_id}_0",
                    chunk_text=stripped,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=position_range,
                    token_count=count_tokens(stripped),
                )
            ]

        # Build paragraph char ranges
        para_ranges: list[tuple[int, int]] = []
        prev_end = 0
        for m in matches:
            para_ranges.append((prev_end, m.start()))
            prev_end = m.end()
        # Last paragraph
        para_ranges.append((prev_end, len(full_text)))

        # Build chunks from non-empty paragraphs
        chunks: list[Chunk] = []
        chunk_index = 0
        for para_start, para_end in para_ranges:
            para_text = full_text[para_start:para_end].strip()
            if not para_text:
                continue

            position_range = self._position_from_char_range(
                para_start, para_end, unit_char_ranges, raw_units
            )

            chunk_id = f"section_aware_{doc_id}_{chunk_index}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=para_text,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=position_range,
                    token_count=count_tokens(para_text),
                )
            )
            chunk_index += 1

        return chunks

    def _build_full_position_range(self, raw_units: list[RawUnit]) -> PositionMetadata:
        """Build a position metadata range covering all raw_units."""
        if not raw_units:
            return PositionMetadata()

        first_meta = raw_units[0].position_metadata
        last_meta = raw_units[-1].position_metadata

        return self._merge_position_metadata(first_meta, last_meta)

    def _merge_position_metadata(
        self, first: PositionMetadata, last: PositionMetadata
    ) -> PositionMetadata:
        """Merge two PositionMetadata objects into a range spanning both."""
        return PositionMetadata(
            line_number_start=first.line_number_start,
            line_number_end=last.line_number_end
            if last.line_number_end is not None
            else last.line_number_start,
            page_number_start=first.page_number_start,
            page_number_end=last.page_number_end
            if last.page_number_end is not None
            else last.page_number_start,
            item_id=first.item_id
            if first.item_id == last.item_id
            else (
                f"{first.item_id}-{last.item_id}"
                if first.item_id and last.item_id
                else first.item_id or last.item_id
            ),
        )
