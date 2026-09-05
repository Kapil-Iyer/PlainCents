"""
ML-G evaluation-protocol and artifact-integrity tests.

These guard the claims the product makes about its own categorizer. A model
that is fine but is described wrongly -- or that ships with a recipe nobody
selected -- is a correctness problem, not a documentation problem.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from backend.services.ambiguity import is_structurally_ambiguous
from ml.categorization.candidates_v2 import SparseTextCandidate, rebuild_candidate
from ml.categorization.run_mlg_bakeoff import (
    ABSTAIN_CATEGORY,
    FINAL_TEST,
    SELECTION_RECORD_PATH,
    TRAIN,
    VALIDATION,
    apply_policy,
    get_or_build_split,
    load_benchmark,
    top_margin,
)
from ml.categorization.text_normalize_v2 import (
    normalize_deployment_text_v2,
    resolve_normalizer,
)
from ml.common.splitting import verify_split_isolation

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# -- corpus and split integrity ----------------------------------------------


def test_split_isolates_merchant_groups_across_every_partition():
    """The whole benchmark is meaningless without this: if a merchant group
    straddles TRAIN and FINAL_TEST, the reported score measures memorization,
    not generalization to an unseen merchant."""
    df = load_benchmark()
    report = verify_split_isolation(get_or_build_split(df).apply(df), "merchant_group")

    assert report["all_intersections_empty"], report["intersections"]
    for partition in (TRAIN, VALIDATION, FINAL_TEST):
        assert report["row_counts"][partition] > 0
        assert report["unique_merchant_groups"][partition] > 0


def test_corpus_shares_head_nouns_across_merchant_groups():
    """The v1 corpus gave each head noun exactly one merchant group, so a
    held-out group shared no feature with anything in TRAIN. Head-noun
    redundancy is the property that makes generalization possible at all --
    if a future edit removes it, the corpus silently reverts to v1's failure
    mode with no test failing anywhere else."""
    df = load_benchmark()
    groups = df.loc[~df["is_ambiguous"]].drop_duplicates("merchant_group")

    shared = 0
    for category, rows in groups.groupby("true_category"):
        tokens_per_group = [set(str(g).split()) for g in rows["merchant_group"]]
        for i, tokens in enumerate(tokens_per_group):
            others = set().union(*(t for j, t in enumerate(tokens_per_group) if j != i)) \
                if len(tokens_per_group) > 1 else set()
            if tokens & others:
                shared += 1
        del category

    assert shared / len(groups) > 0.5, (
        "fewer than half of merchant groups share a word with another group in "
        "their own category -- head-noun redundancy has been lost"
    )


def test_no_category_label_leaks_into_any_description():
    """A description containing its own label would make the benchmark
    trivially solvable and the reported numbers meaningless."""
    from config import CATEGORIES

    df = load_benchmark()
    haystack = " ".join(df["merchant"].tolist()).upper()
    for category in CATEGORIES:
        if category == "Other":
            continue  # "OTHER" is not a word this corpus uses; checked below
        assert category.upper() not in haystack, category
    assert "RENT & UTILITIES" not in haystack


# -- decision policy ----------------------------------------------------------


def test_zero_feature_rows_always_abstain_regardless_of_threshold():
    """The core ML-F failure: an all-zero feature row makes a linear model
    return argmax(intercept_) -- one fixed class for every evidence-free
    input. That must never be served, at any threshold."""
    preds = np.array(["Food & Dining", "Transport"], dtype=object)
    scores = np.array([[0.9, 0.1], [0.9, 0.1]])
    n_active = np.array([0, 5])

    out = apply_policy(preds, scores, n_active, min_margin=0.0)

    assert out[0] == ABSTAIN_CATEGORY
    assert out[1] == "Transport"


def test_low_margin_rows_abstain_and_confident_rows_do_not():
    preds = np.array(["Food & Dining", "Transport"], dtype=object)
    scores = np.array([[0.34, 0.33, 0.33], [0.80, 0.15, 0.05]])
    n_active = np.array([10, 10])

    out = apply_policy(preds, scores, n_active, min_margin=0.05)

    assert out[0] == ABSTAIN_CATEGORY  # margin 0.01
    assert out[1] == "Transport"       # margin 0.65


def test_top_margin_is_top_minus_runner_up():
    scores = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    assert top_margin(scores) == pytest.approx([0.2, 0.0])


# -- structural ambiguity: coverage AND over-routing --------------------------

def test_ambiguity_rule_catches_every_ambiguous_row_without_over_routing():
    """Both directions matter. Coverage alone was already 100% before ML-G;
    what was broken was the false-positive side, which routed 13.8% of
    legitimate FINAL_TEST rows to Other."""
    df = load_benchmark()

    ambiguous = df[df["is_ambiguous"]]
    routed = ambiguous["merchant"].map(is_structurally_ambiguous)
    assert routed.all(), ambiguous.loc[~routed, "merchant"].tolist()

    normal = df[~df["is_ambiguous"]]
    false_positives = normal["merchant"].map(is_structurally_ambiguous)
    assert not false_positives.any(), normal.loc[false_positives, "merchant"].head(10).tolist()


# -- normalizer registry ------------------------------------------------------


def test_normalizer_resolves_by_name_and_refuses_unknown_names():
    """Silently falling back to no normalization on an unknown name is
    exactly the train/serve skew the registry exists to prevent."""
    assert resolve_normalizer("normalize_deployment_text_v2") is normalize_deployment_text_v2
    assert resolve_normalizer(None) is None
    with pytest.raises(ValueError):
        resolve_normalizer("not_a_real_normalizer")


def test_normalizer_strips_boilerplate_but_never_merchant_words():
    assert normalize_deployment_text_v2(
        "VISA DEBIT PURCHASE - 4821 CAREWELL PHARMACY") == "CAREWELL PHARMACY"
    assert normalize_deployment_text_v2(
        "E-TRANSFER SENT MAPLEWOOD DINER REF44120") == "MAPLEWOOD DINER"
    assert normalize_deployment_text_v2("ONLINE PURCHASE STREAMBOXPLUSCOM") == "STREAMBOXPLUSCOM"
    # Nothing but boilerplate -> empty, which callers read as "names nothing".
    assert normalize_deployment_text_v2("E-TRANSFER SENT") == ""
    assert normalize_deployment_text_v2("") == ""


# -- production artifact integrity -------------------------------------------


@pytest.mark.skipif(not SELECTION_RECORD_PATH.exists(),
                    reason="ML-G selection record not present")
def test_production_artifact_matches_the_frozen_selection():
    """The shipped artifact must BE the selected recipe, with the selected
    decision policy. ML-F shipped a model whose recipe the service could not
    actually serve (it ignored the normalizer entirely); nothing detected
    that, because nothing compared the two."""
    from config import CATEGORIZER_MODEL_PATH

    if not Path(CATEGORIZER_MODEL_PATH).exists():
        pytest.skip("production artifact not built in this environment")

    import joblib

    selection = json.loads(SELECTION_RECORD_PATH.read_text(encoding="utf-8"))
    winner_cfg = selection["winner"]["exact_configuration"]
    policy = selection["abstention_policy"]

    payload = joblib.load(CATEGORIZER_MODEL_PATH)

    assert payload["normalizer_name"] == winner_cfg["normalizer_name"]
    assert payload["min_margin"] == policy["min_margin"]
    assert payload["abstain_category"] == policy["abstain_category"]
    recipe = payload["metadata"]["recipe"]
    assert recipe["model_kind"] == winner_cfg["model_kind"]
    assert recipe["C"] == winner_cfg["C"]
    assert recipe["class_weight"] == winner_cfg["class_weight"]


@pytest.mark.skipif(not SELECTION_RECORD_PATH.exists(),
                    reason="ML-G selection record not present")
def test_selection_record_reports_a_sealed_final_test_evaluated_once():
    selection = json.loads(SELECTION_RECORD_PATH.read_text(encoding="utf-8"))

    assert "sealed_final_test_model_only" in selection
    assert "sealed_final_test_with_policy" in selection
    # Sanity floor, well below the achieved score: this exists to catch a
    # catastrophic regression (e.g. the corpus or split being rebuilt wrong),
    # not to pin an exact number that legitimate work may move.
    assert selection["sealed_final_test_model_only"]["macro_f1"] > 0.40
    assert selection["dataset"]["dataset_id"] == "deployment_benchmark_v2"


def test_rebuild_candidate_round_trips_through_json():
    """The production build reconstructs the winner from a JSON record, where
    ngram_range tuples have become lists. sklearn requires real tuples."""
    original = SparseTextCandidate(
        name="x",
        word_config={"ngram_range": (1, 2), "max_features": None, "sublinear_tf": True},
        char_config={"analyzer": "char_wb", "ngram_range": (2, 6), "max_features": 100},
        normalizer_name="normalize_deployment_text_v2",
    )
    round_tripped = rebuild_candidate(json.loads(json.dumps(original.config())))

    assert round_tripped.word_config["ngram_range"] == (1, 2)
    assert round_tripped.char_config["ngram_range"] == (2, 6)
    assert round_tripped.normalizer_name == original.normalizer_name
