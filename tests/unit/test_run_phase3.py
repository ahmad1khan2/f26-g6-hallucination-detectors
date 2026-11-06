"""Pure parts of scripts/run_phase3.py: language order, size planning, round-robin slices, accuracy warnings."""
import yaml

from scripts import run_phase3 as rp

COUNTS = {"mtrex": 100, "xfactr": 50, "psiloqa": 20}
LANGUAGES = {"mtrex": 5, "xfactr": 8, "psiloqa": 5}
THROUGHPUT = {"mam": dict.fromkeys(COUNTS, 1.0), "clf": dict.fromkeys(COUNTS, 0.0),
              "sampling": dict.fromkeys(COUNTS, 2.0)}  # MAM alone: 1,000 s; sampling everything: 2,000 s


def test_mtrex_languages_run_in_priority_order():
    assert rp.languages("mtrex", {}) == ["en", "hi", "de", "bn", "ur"]


def test_xfactr_starts_with_english_then_the_chosen_languages_in_file_order():
    assert rp.languages("xfactr", {}) == ["en", "ja", "zh", "vi", "tl", "sw", "yo", "ilo"]


def test_psiloqa_languages_follow_data_yaml():
    assert rp.languages("psiloqa", {}) == ["en", "hi", "zh", "fi", "ar"]


def test_the_languages_key_restricts_but_keeps_priority_order():
    cfg = {"languages": {"mtrex": ["ur", "en"]}}
    assert rp.languages("mtrex", cfg) == ["en", "ur"]
    assert rp.languages("xfactr", cfg) == rp.languages("xfactr", {})


def test_everything_fits_gives_no_sampling_limit():
    assert rp.plan_sizes(THROUGHPUT, COUNTS, 1.0, LANGUAGES) == {"mtrex": None, "xfactr": None, "psiloqa": None}


def test_partial_fit_gives_one_n_for_every_dataset_capped_by_its_own_count():
    # budget 800 s; 36 s per item while all three datasets grow, 26 s once psiloqa is exhausted at 20 items
    assert rp.plan_sizes(THROUGHPUT, COUNTS, 0.5, LANGUAGES) == {"mtrex": 23, "xfactr": 23, "psiloqa": None}


def test_mam_alone_exceeding_the_hours_gives_zero_and_a_warning(capsys):
    assert rp.plan_sizes(THROUGHPUT, COUNTS, 0.2, LANGUAGES) == {"mtrex": 0, "xfactr": 0, "psiloqa": 0}
    assert "MAM" in capsys.readouterr().out


def test_classifier_time_is_never_cut():
    throughput = THROUGHPUT | {"clf": dict.fromkeys(COUNTS, 100.0)}  # +1,800 s over 18 language folders
    assert rp.plan_sizes(throughput, COUNTS, 0.7, LANGUAGES) == {"mtrex": 0, "xfactr": 0, "psiloqa": 0}


def test_accuracy_within_ten_points_of_the_paper_gives_no_warning():
    assert rp.accuracy_warnings({"capitals": 74.0, "country": 87.0, "official_language": 60.0}) == []


def test_accuracy_more_than_ten_points_off_warns_naming_the_category():
    (warning,) = rp.accuracy_warnings({"capitals": 74.0, "country": 87.5, "official_language": 70.0})
    assert "country" in warning and "77" in warning


def test_category_accuracy_is_the_share_of_label_zero_per_relation(mtrex_records):
    answers = [{"uid": r.uid, "label": label} for r, label in zip(mtrex_records, (0, 1, 0))]
    assert rp.category_accuracy(answers, mtrex_records) == {"capitals": 200 / 3}


def test_round_robin_gives_each_language_a_slice_per_turn_and_keeps_groups_whole():
    groups = {"en": [["a1", "a2"], ["b"], ["c"]], "hi": [["d"]]}
    assert list(rp.round_robin(groups, 2)) == [("en", ["a1", "a2", "b"]), ("hi", ["d"]), ("en", ["c"])]


def test_full_refuses_without_measured_and_never_loads_the_model(tmp_path, capsys):
    config = tmp_path / "run.yaml"
    config.write_text(yaml.safe_dump({"mam_limit": None}), encoding="utf-8")

    def loader():
        raise AssertionError("the model must not load")

    code = rp.main(["full", "--config", str(config), "--results", str(tmp_path / "r")], loader=loader)

    assert code != 0 and "run timing first" in capsys.readouterr().out
