# -*- coding: utf-8 -*-
"""
Spotlight / exception report builder.
"""

from datetime import date
from typing import List

from .schemas import (
    SpotlightReportData,
    ExceptionDefinition,
    SpotlightAnomaly,
)


def build_spotlight_data(kb, top_n: int = 5) -> SpotlightReportData:
    """
    Build SpotlightReportData from spotlight segments.
    Rank by exposure * |A/E - 1| and take Top N.
    """
    exception_def = ExceptionDefinition(
        ae_deviation_threshold=0.15,
        min_exposure=1000,
        notes="Top anomalies ranked by exposure × |A/E−1|. Spotlight segments generated from semantic pipeline.",
    )

    rows = []
    for seg in kb.spotlight_details.values():
        stats = seg.get("statistics", {})
        exposure = stats.get("exposure")
        ae = stats.get("ae_ratio")
        if exposure is None or ae is None:
            continue
        score = float(exposure) * abs(float(ae) - 1.0)
        rows.append((score, seg, stats))

    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]

    anomalies: List[SpotlightAnomaly] = []
    for score, seg, stats in rows:
        credibility_raw = stats.get("credibility_score")
        credibility = float(credibility_raw) if credibility_raw is not None else None

        # Extract structured stability metrics
        stab_dict = stats.get("stability", {})
        stability_score = None
        persistence = None
        log_ae_sd = None
        stability_note_str = None

        if isinstance(stab_dict, dict) and stab_dict:
            stability_score_raw = stab_dict.get("stability_score")
            stability_score = float(stability_score_raw) if stability_score_raw is not None else None
            persistence_raw = stab_dict.get("persistence")
            persistence = float(persistence_raw) if persistence_raw is not None else None
            log_ae_sd_raw = stab_dict.get("log_ae_sd")
            log_ae_sd = float(log_ae_sd_raw) if log_ae_sd_raw is not None else None
            anomaly_years = stab_dict.get("anomaly_years", "?")
            years_analyzed = stab_dict.get("years_analyzed", "?")
            direction = stab_dict.get("direction", "?")
            stability_note_str = (
                f"Anomaly present in {anomaly_years}/{years_analyzed} years "
                f"(direction: {direction})"
            )
        else:
            stability_note_str = stats.get(
                "stability_note",
                "Stability metrics not computed at this segment level.",
            )

        # Compute deaths delta (actual - expected): positive = excess mortality
        actual_d = stats.get("actual_deaths", stats.get("observed_deaths"))
        expected_d = stats.get("expected_deaths")
        deaths_delta = None
        if actual_d is not None and expected_d is not None:
            deaths_delta = round(float(actual_d) - float(expected_d), 1)

        anomalies.append(
            SpotlightAnomaly(
                segment_id=seg["segment_id"],
                rule_text=seg.get("rule_text", ""),
                ae_ratio=float(stats.get("ae_ratio")),
                exposure=float(stats.get("exposure")),
                deaths=actual_d,
                expected_deaths=expected_d,
                credibility_score=credibility,
                stability_score=stability_score,
                persistence=persistence,
                log_ae_sd=log_ae_sd,
                stability_note=stability_note_str,
                deaths_delta=deaths_delta,
            )
        )

    return SpotlightReportData(
        report_date=str(date.today()),
        exception_definition=exception_def,
        top_anomalies=anomalies,
        disclaimers=[
            "Spotlight segments are intended for exception detection and hypothesis generation.",
            "Ranking uses exposure × |A/E−1| to prioritize segments with both material deviation and scale.",
        ],
    )
