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


GLOSSARY = [
    "Карл Поппер",
    "Томас Кун",
    "Бас ван Фраассен",
    "Вильгельм Гумбольдт",
    "фальсификационизм",
    "наука",
]


def test_a_run_of_glossary_terms_is_dropped() -> None:
    from examprep.clean import is_glossary_echo

    echo = "Бас ван Фраассен, Вильгельм Гумбольдт, Карл Поппер"
    assert is_glossary_echo(echo, GLOSSARY)


def test_a_sentence_mentioning_one_name_is_kept() -> None:
    from examprep.clean import is_glossary_echo

    normal = "Карл Поппер предложил считать критерием научности возможность опровержения."
    assert not is_glossary_echo(normal, GLOSSARY)


def test_two_names_inside_a_real_sentence_are_kept() -> None:
    from examprep.clean import is_glossary_echo

    normal = (
        "Томас Кун спорит с тем, как Карл Поппер описывает развитие науки, "
        "и вводит понятие парадигмы вместо последовательной серии опровержений."
    )
    assert not is_glossary_echo(normal, GLOSSARY)


def test_clean_segments_drops_the_echo() -> None:
    segments = [
        seg("Бас ван Фраассен, Вильгельм Гумбольдт, Карл Поппер", 0),
        seg("Сегодня мы начнём с научной революции.", 5),
    ]
    cleaned = clean_segments(segments, glossary=GLOSSARY)

    assert [s.text for s in cleaned] == ["Сегодня мы начнём с научной революции."]


def test_subtitle_credits_with_initials_are_dropped() -> None:
    segments = [
        seg(".Семкин Корректор А.Егорова", 0),
        seg("Редактор М.Иванова", 5),
        seg("Сегодня говорим о Платоне.", 10),
    ]
    assert [s.text for s in clean_segments(segments)] == ["Сегодня говорим о Платоне."]


def test_an_editor_without_initials_is_kept() -> None:
    """A lecture may well discuss editors; only the credits pattern goes."""

    segments = [seg("Редактор журнала настаивал на публикации статьи.", 0)]
    assert len(clean_segments(segments)) == 1
