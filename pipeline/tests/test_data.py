"""The hand-written files under data/ must always satisfy the schema."""

from examprep import store
from examprep.validate import validate_data


def test_question_sets_are_present_and_valid() -> None:
    report = validate_data()
    assert report.ok, report.errors
    assert len(report.checked) >= 3


def test_expected_question_counts() -> None:
    sets = {p.stem: store.load_question_set(p) for p in store.question_set_paths()}
    assert len(sets["mipt-hps-general-2025-26"].questions) == 26
    assert len(sets["skvorchevsky-2025-26"].questions) == 21


def test_crosswalk_covers_only_known_questions() -> None:
    known = {q.id for p in store.question_set_paths() for q in store.load_question_set(p).questions}
    for path in store.crosswalk_paths():
        crosswalk = store.load_crosswalk(path)
        for pair in crosswalk.pairs:
            assert pair.course in known
            assert set(pair.general) <= known
