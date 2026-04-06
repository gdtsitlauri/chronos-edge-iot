from __future__ import annotations

from chronos.evaluation.ablation import AblationStudy


def test_ablation_relative_change_is_bounded():
    study = AblationStudy(base_config={})
    study.record_variant_results("CHRONOS-full", {"avg_combined_reward": 0.000001})
    study.record_variant_results("CHRONOS-noSNN", {"avg_combined_reward": -100.0})

    impacts = study.compute_ablation_impacts()
    pct = impacts["CHRONOS-noSNN"]["impacts"]["avg_combined_reward"]["relative_change_pct"]

    assert -200.0 <= pct <= 200.0


def test_ablation_report_mentions_symmetric_percentage_note():
    study = AblationStudy(base_config={})
    study.record_variant_results("CHRONOS-full", {"avg_combined_reward": 1.0})
    study.record_variant_results("CHRONOS-noSNN", {"avg_combined_reward": 0.5})

    report = study.generate_report()

    assert "symmetric relative change" in report
