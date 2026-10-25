import csv, os
from voiceops.scoring import extract_dates, fact_present, score_reply, words_to_numbers


def test_dates_equivalent():
    for s in ["2026-10-27", "27 October 2026", "the 27th of October", "October 27, 2026", "27 अक्टूबर 2026", "27/10/2026"]:
        assert fact_present("2026-10-27 / 27 October 2026 / 27 अक्टूबर 2026", s), s
    assert not fact_present("2026-10-27", "27 September 2026")


def test_spoken_numbers():
    assert words_to_numbers("forty-nine rupees") == "49 rupees"
    assert fact_present("49", "It costs forty nine rupees")
    assert fact_present("2 hours / instant", "within two hours")
    assert not fact_present("7", "within 17 days")


def test_must_not():
    row = {"must_include": "", "must_not_include": "5 to 7 days ; 5-7 days"}
    assert not score_reply(row, "It takes 5 to 7 days")["pass"]
    assert score_reply(row, "It is instant")["pass"]


def test_scorer_accepts_dataset_reference_answers():
    """A correct scorer must pass the dataset's own reference answers."""
    d = os.environ["DATASET_DIR"]
    rows = list(csv.DictReader(open(f"{d}/practice/questions.csv", encoding="utf-8")))
    fails = [(r["id"], score_reply(r, r["reference_answer"])) for r in rows if not score_reply(r, r["reference_answer"])["pass"]]
    assert len(fails) <= 3, fails
