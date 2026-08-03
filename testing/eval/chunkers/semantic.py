"""Semantic embedding-based chunker.

Groups consecutive sentences by cosine similarity of their embeddings,
starting a new chunk when similarity drops below a threshold or the
sentence count cap is reached. Falls back to fixed batches of 5 if
the embedding API is unavailable.
"""

from testing.eval import config
from testing.eval.chunkers.base import BaseChunker
from testing.eval.embeddings import cosine_similarity, embed_texts
from testing.eval.models import Chunk, PositionMetadata, RawUnit, count_tokens


class SemanticChunker(BaseChunker):
    """Chunk text by grouping semantically similar consecutive sentences."""

    def __init__(
        self,
        threshold: float = config.SEMANTIC_THRESHOLD,
        max_sentences: int = config.MAX_SENTENCES_PER_CHUNK,
    ):
        self.threshold = threshold
        self.max_sentences = max_sentences

    @property
    def strategy_name(self) -> str:
        return "semantic"

    def chunk(self, raw_units: list[RawUnit]) -> list[Chunk]:
        """Split raw units into semantically coherent chunks.

        Algorithm:
        1. Concatenate all raw_unit text, preserving character-to-raw_unit mapping.
        2. Split into sentences at ". ", "? ", "! " boundaries.
        3. Embed sentences and group by cosine similarity above threshold.
        4. Fall back to fixed batches of 5 if embedding fails.
        5. Build Chunk objects with position metadata from contributing raw units.
        """
        if not raw_units:
            return []

        # Derive doc_id from the first raw unit (all should share the same doc)
        doc_id = raw_units[0].doc_id

        # Step 1: Concatenate text and build character-to-raw_unit index mapping
        # Each entry in char_to_unit maps a character offset to the raw_unit index
        full_text = ""
        char_to_unit: list[int] = []  # char offset -> raw_unit index

        for unit_idx, unit in enumerate(raw_units):
            if full_text:
                # Add a space separator between units
                full_text += " "
                char_to_unit.append(unit_idx)
            full_text += unit.text
            char_to_unit.extend([unit_idx] * len(unit.text))

        if not full_text.strip():
            return []

        # Step 2: Split into sentences using ". ", "? ", "! " boundaries
        # Keep punctuation with the preceding sentence
        sentences, sentence_char_starts = self._split_sentences(full_text)

        if not sentences:
            return []

        # Step 3: Determine which raw_unit each sentence belongs to
        sentence_units = self._map_sentences_to_units(
            sentences, sentence_char_starts, char_to_unit
        )

        # Step 4: Try to embed sentences and group by similarity
        try:
            embeddings = embed_texts(sentences)
            groups = self._group_by_similarity(embeddings)
        except Exception:
            # Step 6: Fall back to fixed batches of 5
            groups = self._group_fixed_batches(len(sentences), batch_size=5)

        # Step 7: Build Chunk objects from sentence groups
        chunks: list[Chunk] = []
        for index, group in enumerate(groups):
            group_sentences = [sentences[i] for i in group]
            chunk_text = " ".join(group_sentences)

            # Determine position metadata range from contributing raw units
            contributing_unit_indices: set[int] = set()
            for sent_idx in group:
                contributing_unit_indices.update(sentence_units[sent_idx])

            position_metadata_range = self._compute_position_range(
                raw_units, contributing_unit_indices
            )

            token_count = count_tokens(chunk_text)
            chunk_id = f"semantic_{doc_id}_{index}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    doc_id=doc_id,
                    strategy=self.strategy_name,
                    position_metadata_range=position_metadata_range,
                    token_count=token_count,
                )
            )

        return chunks

    def _split_sentences(self, text: str) -> tuple[list[str], list[int]]:
        """Split text into sentences at '. ', '? ', '! ' boundaries.

        Punctuation is kept with the preceding sentence.

        Returns:
            Tuple of (sentences list, character start offsets list).
        """
        sentences: list[str] = []
        starts: list[int] = []
        boundaries = [". ", "? ", "! "]

        current_start = 0
        i = 0
        while i < len(text):
            matched = False
            for boundary in boundaries:
                if text[i : i + len(boundary)] == boundary:
                    # Include the punctuation character with the sentence
                    sentence = text[current_start : i + 1].strip()
                    if sentence:
                        sentences.append(sentence)
                        starts.append(current_start)
                    current_start = i + len(boundary)
                    i = current_start
                    matched = True
                    break
            if not matched:
                i += 1

        # Add remaining text as a final sentence
        remaining = text[current_start:].strip()
        if remaining:
            sentences.append(remaining)
            starts.append(current_start)

        return sentences, starts

    def _map_sentences_to_units(
        self,
        sentences: list[str],
        sentence_char_starts: list[int],
        char_to_unit: list[int],
    ) -> list[set[int]]:
        """Map each sentence to the set of raw_unit indices it spans.

        Uses the character start offset and sentence length to find all
        raw_unit indices covered by the sentence's character span.

        Returns:
            List where each element is a set of raw_unit indices that
            the corresponding sentence covers.
        """
        sentence_units: list[set[int]] = []

        for idx, sentence in enumerate(sentences):
            start = sentence_char_starts[idx]
            units: set[int] = set()

            # Scan through characters in the sentence's span to collect
            # all contributing raw_unit indices
            end = min(start + len(sentence) + 10, len(char_to_unit))
            for c in range(start, end):
                if c < len(char_to_unit):
                    units.add(char_to_unit[c])

            sentence_units.append(units)

        return sentence_units

    def _group_by_similarity(
        self, embeddings: list[list[float]]
    ) -> list[list[int]]:
        """Group consecutive sentence indices by cosine similarity.

        A new group starts when similarity between adjacent sentences
        drops below the threshold, or the group reaches max_sentences.
        """
        if not embeddings:
            return []

        groups: list[list[int]] = []
        current_group: list[int] = [0]

        for i in range(1, len(embeddings)):
            similarity = cosine_similarity(embeddings[i - 1], embeddings[i])

            if (
                similarity > self.threshold
                and len(current_group) < self.max_sentences
            ):
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]

        # Don't forget the last group
        if current_group:
            groups.append(current_group)

        return groups

    def _group_fixed_batches(
        self, num_sentences: int, batch_size: int = 5
    ) -> list[list[int]]:
        """Group sentence indices into fixed-size batches (fallback)."""
        groups: list[list[int]] = []
        for i in range(0, num_sentences, batch_size):
            groups.append(list(range(i, min(i + batch_size, num_sentences))))
        return groups

    def _compute_position_range(
        self, raw_units: list[RawUnit], unit_indices: set[int]
    ) -> PositionMetadata:
        """Compute the position metadata range spanning all contributing units."""
        if not unit_indices:
            return PositionMetadata()

        sorted_indices = sorted(unit_indices)
        first_unit = raw_units[sorted_indices[0]]
        last_unit = raw_units[sorted_indices[-1]]

        first_meta = first_unit.position_metadata
        last_meta = last_unit.position_metadata

        result = PositionMetadata()

        # Line number range (for .md/.txt)
        if first_meta.line_number_start is not None:
            result.line_number_start = first_meta.line_number_start
            result.line_number_end = (
                last_meta.line_number_end
                if last_meta.line_number_end is not None
                else last_meta.line_number_start
            )

        # Page number range (for .pdf)
        if first_meta.page_number_start is not None:
            result.page_number_start = first_meta.page_number_start
            result.page_number_end = (
                last_meta.page_number_end
                if last_meta.page_number_end is not None
                else last_meta.page_number_start
            )

        # Item ID (for .html) — use first unit's item_id
        if first_meta.item_id is not None:
            result.item_id = first_meta.item_id

        return result
