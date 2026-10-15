import math

import pytest

from biomedrag.eval import metrics


def test_first_gold_rank_finds_the_first_hit_or_none():
    assert metrics.first_gold_rank(['a', 'b', 'c'], ['c']) == 3
    assert metrics.first_gold_rank(['a', 'b', 'c'], ['x', 'b', 'c']) == 2
    assert metrics.first_gold_rank(['a', 'b'], ['z']) is None
    assert metrics.first_gold_rank([], ['z']) is None


def test_hits_and_recall_at_k():
    ranks = [1, 3, None, 7]
    assert metrics.hits(ranks, 1) == [1, 0, 0, 0]
    assert metrics.hits(ranks, 5) == [1, 1, 0, 0]
    assert metrics.mean(metrics.hits(ranks, 10)) == 0.75


def test_reciprocal_rank_and_cutoff():
    ranks = [1, 2, 4, None, 12]
    assert metrics.reciprocal_ranks(ranks, 10) == [1.0, 0.5, 0.25, 0.0, 0.0]
    assert metrics.mean(metrics.reciprocal_ranks([1, 2], 10)) == 0.75


def test_percentile_and_mean_of_empty():
    assert metrics.percentile([1, 2, 3, 4, 5], 50) == 3
    assert metrics.percentile([10, 20, 30, 40, 50], 95) == pytest.approx(48)
    assert math.isnan(metrics.mean([])) and math.isnan(metrics.percentile([], 50))


def test_bootstrap_interval_contains_the_mean_and_is_repeatable():
    values = [1.0] * 30 + [0.0] * 20
    lo, hi = metrics.bootstrap_ci(values)
    assert lo < 0.6 < hi and 0.4 < lo and hi < 0.8
    assert (lo, hi) == metrics.bootstrap_ci(values)           # same seed, same answer


def test_more_questions_give_a_narrower_interval():
    small = metrics.bootstrap_ci([1.0, 0.0] * 10)
    large = metrics.bootstrap_ci([1.0, 0.0] * 200)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_interval_of_constant_values_is_that_value():
    assert metrics.bootstrap_ci([0.7] * 10) == (pytest.approx(0.7), pytest.approx(0.7))


def test_paired_bootstrap_detects_a_clear_difference():
    better, worse = [1.0] * 40 + [0.0] * 10, [0.0] * 30 + [1.0] * 10 + [0.0] * 10
    diff, lo, _ = metrics.paired_bootstrap(better, worse)
    assert diff == pytest.approx(0.6) and lo > 0


def test_paired_bootstrap_of_identical_systems_is_zero():
    same = [1.0, 0.0, 1.0, 1.0]
    assert metrics.paired_bootstrap(same, same) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_needs_matching_questions():
    with pytest.raises(ValueError, match='same questions'):
        metrics.paired_bootstrap([1.0], [1.0, 0.0])


def test_sweep_counts_answered_and_refused_at_a_cutoff():
    rows = metrics.sweep_thresholds([5, 4, 3, 2], [1, 2.5, 0, None])
    at3 = next(r for r in rows if r['threshold'] == 3)
    assert at3['answered'] == 0.75 and at3['refused'] == 1.0     # 5,4,3 answered; all four unanswerable < 3
    assert rows[0]['answered'] == 1.0 and rows[0]['refused'] == 0.25     # lowest cutoff: only None is refused
    assert rows[-1]['answered'] == 0.0 and rows[-1]['refused'] == 1.0    # refuse everything


def test_a_question_with_no_score_is_always_refused():
    rows = metrics.sweep_thresholds([None, 5.0], [None])
    assert rows[0]['answered'] == 0.5 and rows[0]['refused'] == 1.0


def test_sweep_of_nothing_is_empty():
    assert metrics.sweep_thresholds([None], [None]) == []


def test_choose_threshold_meets_targets_with_the_best_total():
    rows = metrics.sweep_thresholds([9, 8, 7, 6, 5, 4, 3, 2, 1.5, 1.2], [0, 0.5, 1, 2.2, 2.4, 3.5, 5, 0.1, 0.2, 0.3])
    best, met = metrics.choose_threshold(rows, min_refuse=0.7, min_answer=0.7)
    assert met and best['answered'] >= 0.7 and best['refused'] >= 0.7
    assert best['answered'] + best['refused'] == max(r['answered'] + r['refused'] for r in rows
                                                      if r['answered'] >= 0.7 and r['refused'] >= 0.7)


def test_when_no_cutoff_meets_the_targets_the_best_compromise_is_reported_as_not_met():
    rows = metrics.sweep_thresholds([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])       # indistinguishable
    best, met = metrics.choose_threshold(rows, 0.8, 0.85)
    assert not met and best is not None


def test_choose_from_nothing():
    assert metrics.choose_threshold([]) == (None, False)


def test_ties_prefer_the_lower_threshold_so_fewer_good_questions_are_refused():
    rows = [{'threshold': 1.0, 'answered': 0.9, 'refused': 0.9}, {'threshold': 2.0, 'answered': 0.9, 'refused': 0.9}]
    assert metrics.choose_threshold(rows)[0]['threshold'] == 1.0
