from examprep.answer.extract import timecode
from examprep.answer.synthesize import CitationOut, SynthesisResponse, build_citations, renumber
from examprep.answer.verify import quote_matches, quote_start
from examprep.llm import extract_json, render_prompt, strip_thinking
from examprep.schemas import Chunk, Segment

FRAGMENT = (
    "Поппер считает, что критерием научности является не подтверждаемость, "
    "а принципиальная возможность опровержения, то есть фальсифицируемость."
)


def chunk(chunk_id: str = "abc123:0007", text: str = FRAGMENT) -> Chunk:
    return Chunk(chunk_id=chunk_id, video_id="abc123", start=610.0, end=700.0, text=text)


class TestQuoteMatches:
    def test_exact_quote(self) -> None:
        assert quote_matches("критерием научности является не подтверждаемость", FRAGMENT)

    def test_case_and_punctuation_do_not_matter(self) -> None:
        assert quote_matches("КРИТЕРИЕМ НАУЧНОСТИ, является — не подтверждаемость!", FRAGMENT)

    def test_small_drift_is_tolerated(self) -> None:
        """The model tends to tidy the lecturer's grammar; that is still the quote."""

        assert quote_matches("критерием научности является неподтверждаемость", FRAGMENT)

    def test_invented_quote_is_rejected(self) -> None:
        assert not quote_matches("Кун вводит понятие парадигмы и научной революции", FRAGMENT)

    def test_too_short_to_judge(self) -> None:
        assert not quote_matches("наука", FRAGMENT)

    def test_empty_source(self) -> None:
        assert not quote_matches("критерием научности является", "")


class TestBuildCitations:
    def test_keeps_a_verifiable_citation(self) -> None:
        response = SynthesisResponse(
            answer_md="Тезис [1].",
            coverage="full",
            citations=[CitationOut(chunk_id="abc123:0007", quote="критерием научности является")],
        )
        citations, dropped = build_citations(response, {"abc123:0007": chunk()})

        assert dropped == []
        assert [c.n for c in citations] == [1]
        assert citations[0].video_id == "abc123"
        assert citations[0].start == 610.0

    def test_drops_an_unknown_fragment(self) -> None:
        response = SynthesisResponse(
            answer_md="Тезис [1].",
            coverage="full",
            citations=[CitationOut(chunk_id="нет:0001", quote="критерием научности является")],
        )
        citations, dropped = build_citations(response, {"abc123:0007": chunk()})

        assert citations == []
        assert dropped == [1]

    def test_drops_an_invented_quote(self) -> None:
        response = SynthesisResponse(
            answer_md="Тезис [1].",
            coverage="full",
            citations=[CitationOut(chunk_id="abc123:0007", quote="Кун вводит понятие парадигмы")],
        )
        citations, dropped = build_citations(response, {"abc123:0007": chunk()})

        assert citations == []
        assert dropped == [1]

    def test_numbering_is_dense_after_a_drop(self) -> None:
        chunks = {"abc123:0007": chunk(), "abc123:0008": chunk("abc123:0008")}
        response = SynthesisResponse(
            answer_md="Раз [1], два [2], три [3].",
            coverage="full",
            citations=[
                CitationOut(chunk_id="abc123:0007", quote="критерием научности является"),
                CitationOut(chunk_id="нет:0001", quote="что угодно, чего нет в чанке вообще"),
                CitationOut(chunk_id="abc123:0008", quote="возможность опровержения"),
            ],
        )
        citations, dropped = build_citations(response, chunks)

        assert [c.n for c in citations] == [1, 2]
        assert dropped == [2]


class TestRenumber:
    def test_markers_follow_the_surviving_citations(self) -> None:
        response = SynthesisResponse(
            answer_md="Раз [1], два [2], три [3].",
            coverage="full",
            citations=[
                CitationOut(chunk_id="a:0001"),
                CitationOut(chunk_id="b:0002"),
                CitationOut(chunk_id="c:0003"),
            ],
        )
        assert renumber(response.answer_md, response, dropped=[2]) == "Раз [1], два , три [2]."

    def test_nothing_changes_without_drops(self) -> None:
        response = SynthesisResponse(answer_md="Раз [1].", coverage="full")
        assert renumber("Раз [1].", response, dropped=[]) == "Раз [1]."


class TestLLMHelpers:
    def test_thinking_block_is_removed(self) -> None:
        assert strip_thinking('<think>рассуждаю</think>{"a": 1}') == '{"a": 1}'

    def test_unterminated_thinking_block(self) -> None:
        assert strip_thinking('{"a": 1}<think>оборвалось') == '{"a": 1}'

    def test_json_is_found_inside_prose(self) -> None:
        assert extract_json('Вот ответ: {"relevant": true} — готово') == '{"relevant": true}'

    def test_json_inside_a_fenced_block(self) -> None:
        assert extract_json('```json\n{"relevant": false}\n```') == '{"relevant": false}'

    def test_placeholders_are_filled_without_touching_braces(self) -> None:
        template = 'Вопрос: {question}\nПример: {"a": 1}'
        assert render_prompt(template, question="Поппер") == 'Вопрос: Поппер\nПример: {"a": 1}'


def test_timecode() -> None:
    assert timecode(610.0) == "10:10"
    assert timecode(59.9) == "00:59"


SEGMENTS = [
    Segment(start=600.0, end=608.0, text="Так вот, о чём мы говорили в прошлый раз."),
    Segment(start=608.0, end=618.0, text="Поппер считает, что критерием научности является"),
    Segment(
        start=618.0,
        end=628.0,
        text="не подтверждаемость, а принципиальная возможность опровержения.",
    ),
    Segment(start=628.0, end=640.0, text="Это и называется фальсифицируемостью."),
]


class TestQuoteStart:
    def test_points_at_the_segment_where_the_quote_begins(self) -> None:
        assert quote_start(SEGMENTS, "Поппер считает, что критерием научности", 600.0) == 608.0

    def test_quote_spanning_two_segments(self) -> None:
        """The lecturer's sentence rarely fits one segment."""

        quote = "критерием научности является не подтверждаемость"
        assert quote_start(SEGMENTS, quote, 600.0) == 608.0

    def test_falls_back_when_the_quote_is_not_there(self) -> None:
        assert quote_start(SEGMENTS, "Кун вводит понятие парадигмы", 600.0) == 600.0

    def test_falls_back_on_a_quote_too_short_to_place(self) -> None:
        assert quote_start(SEGMENTS, "наука", 600.0) == 600.0


class TestCitationTimecode:
    def test_citation_points_at_the_sentence_not_the_window(self) -> None:
        window = Chunk(
            chunk_id="abc123:0007",
            video_id="abc123",
            start=600.0,
            end=690.0,
            text=" ".join(s.text for s in SEGMENTS),
        )
        response = SynthesisResponse(
            answer_md="Тезис [1].",
            coverage="full",
            citations=[
                CitationOut(chunk_id="abc123:0007", quote="Поппер считает, что критерием научности")
            ],
        )
        citations, dropped = build_citations(
            response, {"abc123:0007": window}, {"abc123": SEGMENTS}
        )

        assert dropped == []
        assert citations[0].start == 608.0

    def test_without_segments_it_keeps_the_window_start(self) -> None:
        window = Chunk(
            chunk_id="abc123:0007",
            video_id="abc123",
            start=600.0,
            end=690.0,
            text=" ".join(s.text for s in SEGMENTS),
        )
        response = SynthesisResponse(
            answer_md="Тезис [1].",
            coverage="full",
            citations=[
                CitationOut(chunk_id="abc123:0007", quote="Поппер считает, что критерием научности")
            ],
        )
        citations, _ = build_citations(response, {"abc123:0007": window})

        assert citations[0].start == 600.0
