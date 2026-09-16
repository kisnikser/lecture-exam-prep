from examprep.ingest import _entries


def test_entries_of_a_playlist_drops_empty_slots() -> None:
    info = {"_type": "playlist", "entries": [{"id": "a"}, None, {"id": "b"}]}
    assert [e["id"] for e in _entries(info)] == ["a", "b"]


def test_entries_of_a_single_video() -> None:
    info = {"id": "a", "title": "t"}
    assert _entries(info) == [info]
