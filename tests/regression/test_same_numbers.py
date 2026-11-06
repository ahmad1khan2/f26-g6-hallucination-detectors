"""Our adapted functions and the authors' originals give identical numbers on fixed inputs."""
from collections import defaultdict

import numpy as np
import pytest
import torch

from src.detectors import ModelArtifacts_Classifier_g6 as clf
from src.detectors import selfcheckGPT_g6 as sgm
from src.detectors import semanticEntropy_g6 as sem
from src.detectors import shared
from tests.regression.authors import load_authors

PASSAGE = "the cat sat on the mat"
SAMPLES = ["the cat sat", "a dog ran on the mat", "the cat sat on the mat today"]
ANSWERS = ["Paris", "Rome", "Paris", "Rome", "Lima"]
LOG_LIKS = [[-0.5, -1.0], [-2.0], [-0.7, -0.2, -1.1], [-1.5, -0.4], [-3.0]]
CUTOFF = 0.75
SEMANTIC_IDS = [0, 1, 0, 1, 2]
SEQUENCE_LOG_PROBS = [-1.2, -0.5, -2.3, -0.9, -1.7]


def test_unigram_selfcheck_score_matches_the_authors():
    authors = load_authors("selfcheckGPT_g6")
    expected = authors.SelfCheck_Ngram.predict(passage=PASSAGE, sampled_passages=SAMPLES, lang_code="en")
    assert sgm.SelfCheck_Ngram.predict(passage=PASSAGE, sampled_passages=SAMPLES, lang_code="en") == expected


def test_semantic_ids_match_the_authors(fake_encoder):
    authors = load_authors("semanticEntropy_g6")
    ours = sem.get_semantic_ids(ANSWERS, CUTOFF, fake_encoder)
    assert ours == authors.get_semantic_ids(ANSWERS, CUTOFF, fake_encoder) == SEMANTIC_IDS


@pytest.mark.parametrize("agg", ["sum", "sum_normalized", "mean"])
def test_logsumexp_by_id_matches_the_authors(agg):
    authors = load_authors("semanticEntropy_g6")
    assert sem.logsumexp_by_id(SEMANTIC_IDS, SEQUENCE_LOG_PROBS, agg) == \
        authors.logsumexp_by_id(SEMANTIC_IDS, SEQUENCE_LOG_PROBS, agg)


def test_entropy_functions_match_the_authors():
    authors = load_authors("semanticEntropy_g6")
    assert sem.predictive_entropy(SEQUENCE_LOG_PROBS) == authors.predictive_entropy(SEQUENCE_LOG_PROBS)
    assert sem.predictive_entropy_rao(SEQUENCE_LOG_PROBS) == authors.predictive_entropy_rao(SEQUENCE_LOG_PROBS)
    assert sem.cluster_assignment_entropy(SEMANTIC_IDS) == authors.cluster_assignment_entropy(SEMANTIC_IDS)


def authors_entropy_loop(authors, str_responses, log_liks, cutoff, encoder):
    """The entropy section of the authors' answer_question, line for line, on the authors' functions."""
    entropies = defaultdict(list)
    semantic_ids = authors.get_semantic_ids(str_responses, cutoff, encoder)
    entropies['cluster_assignment_entropy'].append(authors.cluster_assignment_entropy(semantic_ids))
    for agg_name, agg_func in zip(['', '_sum'], [np.mean, np.sum]):
        log_liks_agg = [agg_func(log_lik) for log_lik in log_liks]
        entropies['regular_entropy' + agg_name].append(authors.predictive_entropy(log_liks_agg))
        cluster_agg_names = ['', '_sum-normalized', '_sum-normalized-rao', '_cmean']
        cluster_aggs = ['sum', 'sum_normalized', 'sum_normalized', 'mean']
        for cluster_agg_name, cluster_agg in zip(cluster_agg_names, cluster_aggs):
            per_id = authors.logsumexp_by_id(semantic_ids, log_liks_agg, agg=cluster_agg)
            name = 'semantic_entropy' + agg_name + cluster_agg_name
            if cluster_agg_name != '_sum-normalized-rao':
                entropies[name].append(authors.predictive_entropy(per_id))
            else:
                entropies[name].append(authors.predictive_entropy_rao(per_id))
    return entropies


def test_compute_entropies_equals_the_authors_loop_run_by_hand(fake_encoder):
    authors = load_authors("semanticEntropy_g6")
    ours = sem.compute_entropies(ANSWERS, LOG_LIKS, CUTOFF, fake_encoder)
    expected = authors_entropy_loop(authors, ANSWERS, LOG_LIKS, CUTOFF, fake_encoder)
    assert dict(ours) == dict(expected) and len(expected) == 11


def test_classifier_layers_start_from_the_same_weights():
    authors = load_authors("ModelArtifacts_Classifier_g6")
    torch.manual_seed(0)
    theirs = authors.FFHallucinationClassifier(8, 16, 0.5)
    torch.manual_seed(0)
    mine = clf.FFHallucinationClassifier(8, 16, 0.5)
    assert all(torch.equal(a, b) for a, b in zip(theirs.state_dict().values(), mine.state_dict().values()))


def test_seeded_classifier_training_gives_the_authors_auroc(tmp_path):
    authors = load_authors("ModelArtifacts_Classifier_g6")
    rng = np.random.default_rng(0)
    labels = np.array([i % 2 for i in range(60)])
    inputs = (rng.normal(size=(60, 8)) + labels[:, None]).astype(np.float32)

    def train(module):
        torch.manual_seed(7)
        return module.gen_classifier_roc(
            inputs=inputs, correct=labels, artifact_name="a", device="cpu",
            cls_results_file_path=str(tmp_path / "r.txt"), cls_results_dir=str(tmp_path), num_hidden_nodes=16,
            dropout_mlp=0.5, learning_rate=1e-4, weight_decay=1e-2, batch_size=16, seed_val=42)

    assert train(clf) == train(authors)


def test_sampling_in_chunks_of_the_full_size_equals_the_authors_samples(tiny_model, tiny_tokenizer, fake_prompts,
                                                                       mtrex_records):
    authors = load_authors("selfcheckGPT_g6")
    chat = shared.chat_text(mtrex_records[0], tiny_tokenizer, fake_prompts)
    torch.manual_seed(5)
    mine = sgm._sample_chunks(chat, tiny_model, tiny_tokenizer, 4, 6, 4)
    torch.manual_seed(5)
    _, _, theirs, _ = authors.generate_responses(question=chat, num_generations=4, max_new_tokens=6,
                                                 tokenizer=tiny_tokenizer, model=tiny_model, device="cpu",
                                                 lang_code="en")
    assert mine == theirs
