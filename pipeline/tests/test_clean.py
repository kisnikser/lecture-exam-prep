from examprep.clean import apply_replacements, clean_segments, normalize
from examprep.schemas import Segment


def seg(text: str, start: float = 0.0) -> Segment:
    return Segment(start=start, end=start + 1.0, text=text)


def test_drops_subtitle_credits_and_empty_segments() -> None:
    segments = [
        seg("Субтитры сделал DimaTorzok"),
        seg("Продолжение следует..."),
        seg("   "),
        seg("..."),
        seg("Аристотель различает четыре причины."),
    ]
    assert [s.text for s in clean_segments(segments)] == ["Аристотель различает четыре причины."]


def test_collapses_looped_segments_but_keeps_the_first_repeat() -> None:
    segments = [seg("так вот", i) for i in range(5)]
    assert len(clean_segments(segments)) == 2


def test_repeat_counter_resets_on_new_text() -> None:
    segments = [seg("а", 0), seg("а", 1), seg("б", 2), seg("а", 3)]
    assert [s.text for s in clean_segments(segments)] == ["а", "а", "б", "а"]


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  Поппер\n  и   Кун ") == "Поппер и Кун"


def test_replacements_are_case_insensitive_and_whole_word() -> None:
    assert apply_replacements("поппер и попперовский", {"поппер": "Поппер"}) == (
        "Поппер и попперовский"
    )
