# -*- coding: utf-8 -*-
"""
Structured Actuarial Report Generator

Complete demo with:
- Random sampling from ILEC dataset
- Real model prediction + SHAP
- Full knowledge base system prompt
- LLM structured insights
- Report with system prompt appendix
- Timing and cost tracking
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from report_generator.schemas import (
    ActuarialInsights, ReportData, FullReport,
    PredictionData, SHAPData, SegmentData, EvidenceChart, PromptContext
)
from report_generator.llm import create_llm_client
from report_generator.renderer import create_renderer
from report_generator.prompt_builder import get_full_system_prompt, get_prompt_stats
from report_generator.knowledge import get_knowledge_base
from report_generator.predictor import create_predictor
from datetime import date
from .schemas import (
    SpotlightReportData,
    SpotlightPromptContext,
    SpotlightInsights,
    ExceptionDefinition,
    SpotlightAnomaly
)


# =============================================================================
# Test Mode: Set to True to use fake high-risk sample that triggers Spotlight
# =============================================================================
USE_TEST_DATA = True  # Change to False for random real sampling
REPORT_MODE = "spotlight"

# Fake sample designed to trigger Spotlight segment SPOT_364:
# - Duration > 27.5 (30 yrs)
# - Face Amount Band > 250,000 - 499,999 (500k-999k)
# - Attained_Age <= 32.5 (30)
# - Issue_Age <= 27.5 (0, child policy)
# This represents a young person with a long-duration, high face amount policy
TEST_SAMPLE = {
    'Attained_Age': 30,
    'Issue_Age': 0,
    'Duration': 30,
    'Sex': 'M',
    'Smoker_Status': 'NS',
    'Insurance_Plan': 'WL',
    'Face_Amount_Band': '06: 500,000 - 999,999',
    'Preferred_Class': 'NA',
    'SOA_Post_Lvl_Ind': 'NA',
    'SOA_Antp_Lvl_TP': 'N/A (Not Term)',
    'SOA_Guar_Lvl_TP': 'N/A (Not Term)'
}


# =============================================================================
# Cost Estimation (Google Gemini 2.5 Flash Lite pricing)
# =============================================================================
COST_PER_INPUT_TOKEN = 0.1 / 1e6  
COST_PER_OUTPUT_TOKEN = 0.4 / 1e6 


def estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    return len(text) // 3


class CostTracker:
    """Track time and cost for each stage."""
    
    def __init__(self):
        self.stages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def start_stage(self, name: str):
        return {'name': name, 'start': time.time(), 'input_tokens': 0, 'output_tokens': 0}
    
    def end_stage(self, stage: dict, input_text: str = "", output_text: str = ""):
        stage['end'] = time.time()
        stage['duration'] = stage['end'] - stage['start']
        stage['input_tokens'] = estimate_tokens(input_text)
        stage['output_tokens'] = estimate_tokens(output_text)
        stage['cost'] = (
            stage['input_tokens'] * COST_PER_INPUT_TOKEN +
            stage['output_tokens'] * COST_PER_OUTPUT_TOKEN
        )
        self.stages.append(stage)
        self.total_input_tokens += stage['input_tokens']
        self.total_output_tokens += stage['output_tokens']
        return stage
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print("TIMING & COST SUMMARY")
        print("=" * 70)
        
        total_time = sum(s['duration'] for s in self.stages)
        total_cost = sum(s['cost'] for s in self.stages)
        
        print(f"\n{'Stage':<35} {'Time':>10} {'Tokens':>10} {'Cost':>12}")
        print("-" * 70)
        
        for stage in self.stages:
            tokens = stage['input_tokens'] + stage['output_tokens']
            print(f"{stage['name']:<35} {stage['duration']:>9.2f}s {tokens:>9,} ${stage['cost']:>10.6f}")
        
        print("-" * 70)
        print(f"{'TOTAL':<35} {total_time:>9.2f}s {self.total_input_tokens + self.total_output_tokens:>9,} ${total_cost:>10.6f}")


def load_sample():
    """Load sample based on USE_TEST_DATA flag."""
    import pandas as pd
    
    data_path = Path(__file__).parent.parent.parent / 'data' / 'ilec_cleaned.parquet'
    
    if not data_path.exists():
        print(f"⚠ Data file not found: {data_path}")
        return None, None
    
    df = pd.read_parquet(data_path)
    
    if USE_TEST_DATA:
        # Use fake high-risk sample
        sample = TEST_SAMPLE.copy()
        print("    [TEST MODE] Using fake high-risk sample")
    else:
        # Random real sampling
        sample = df.sample(1).iloc[0].to_dict()
    
    return sample, df


def find_matching_segment(input_record: dict, kb) -> dict:
    """Find matching coverage segment based on rules."""
    if not kb.coverage_registry:
        return None
    
    age = input_record.get('Attained_Age', 0)
    duration = input_record.get('Duration', 0)
    plan = input_record.get('Insurance_Plan', 'Term')
    smoker = input_record.get('Smoker_Status', 'NS')
    sex = input_record.get('Sex', 'M')
    
    # Smoker status: NS/S = Known, anything else = Unknown
    smoker_known = smoker in ['NS', 'S']
    is_perm = plan in ['WL', 'UL', 'VUL', 'ULSG']  # Permanent plans
    
    for seg in kb.coverage_registry.get('segments', []):
        rules = seg.get('rule_list', [])
        if not rules:
            continue
            
        match = True
        for rule in rules:
            rule_lower = rule.lower()
            
            # Duration rules
            if 'duration' in rule_lower:
                if '<=' in rule and '>' not in rule:
                    threshold = float(rule.split('<=')[1].strip())
                    if not (duration <= threshold):
                        match = False
                elif '>' in rule and '<=' not in rule:
                    threshold = float(rule.split('>')[1].strip())
                    if not (duration > threshold):
                        match = False
                elif '<' in rule and '>' in rule:  # Range: X < Duration <= Y
                    parts = rule.replace('Duration', '').strip().split('<')
                    lower = float(parts[0].strip())
                    upper = float(parts[1].replace('=', '').strip())
                    if not (lower < duration <= upper):
                        match = False
            
            # Age rules  
            elif 'attained_age' in rule_lower or 'age' in rule_lower:
                if '<=' in rule and '>' not in rule:
                    threshold = float(rule.split('<=')[1].strip())
                    if not (age <= threshold):
                        match = False
                elif '>' in rule and '<=' not in rule:
                    threshold = float(rule.split('>')[1].strip())
                    if not (age > threshold):
                        match = False
                elif '<' in rule and '>' in rule:  # Range
                    parts = rule.split('<')
                    lower = float(parts[0].split('>')[-1].strip())
                    upper = float(parts[1].replace('=', '').replace('Attained_Age', '').strip())
                    if not (lower < age <= upper):
                        match = False
            
            # Smoker status rules
            elif 'smoker' in rule_lower:
                if 'is unknown' in rule_lower:
                    if smoker_known:
                        match = False
                elif 'is known' in rule_lower:
                    if not smoker_known:
                        match = False
            
            # Insurance plan rules
            elif 'insurance plan' in rule_lower or 'plan is' in rule_lower:
                if 'is perm' in rule_lower:
                    if not is_perm:
                        match = False
                elif 'is not perm' in rule_lower:
                    if is_perm:
                        match = False
                elif 'is term' in rule_lower:
                    if plan != 'Term':
                        match = False
                elif 'is not term' in rule_lower:
                    if plan == 'Term':
                        match = False
                elif 'is ulsg' in rule_lower:
                    if plan != 'ULSG':
                        match = False
                elif 'is not ulsg' in rule_lower:
                    if plan == 'ULSG':
                        match = False
            
            # Sex rules
            elif 'sex is' in rule_lower:
                if 'female' in rule_lower and sex != 'F':
                    match = False
                elif 'male' in rule_lower and sex != 'M':
                    match = False
        
        if match:
            return seg
    
    # Fallback: return first segment if no match
    segments = kb.coverage_registry.get('segments', [])
    return segments[0] if segments else None

def find_matching_spotlight(input_record: dict, kb) -> dict:
    """Find matching spotlight segment based on rules (heuristic matching)."""
    if not kb.spotlight_summary:
        return None
    
    anomalies = kb.spotlight_summary.get('anomalies', [])
    if not anomalies:
        return None
    
    age = input_record.get('Attained_Age', 0)
    duration = input_record.get('Duration', 0)
    issue_age = input_record.get('Issue_Age', 0)
    face_band = input_record.get('Face_Amount_Band', '')
    
    # Extract face amount band order for comparison
    face_order = int(face_band.split(':')[0]) if ':' in face_band else 0
    
    # Try to match SPOT_364: Duration > 27.5, Face > 250k (band 05+), Age <= 32.5, Issue <= 27.5
    for anom in anomalies:
        rules = anom.get('rule_list', [])
        match = True
        for rule in rules:
            if 'Duration >' in rule and not (duration > 27.5):
                match = False
            elif 'Attained_Age <=' in rule and not (age <= 32.5):
                match = False
            elif 'Issue_Age <=' in rule and not (issue_age <= 27.5):
                match = False
            elif 'Face Amount Band >' in rule and not (face_order >= 5):
                match = False
        if match and len(rules) > 0:
            return anom
    
    # If no exact match, return the first anomaly for demo purposes
    return anomalies[0] if anomalies else None



def create_report_data(sample: dict, df, predictor, kb) -> ReportData:
    """Create ReportData from sample."""
    
    features = ['Attained_Age', 'Issue_Age', 'Duration', 'Sex', 'Smoker_Status',
                'Insurance_Plan', 'Face_Amount_Band', 'Preferred_Class',
                'SOA_Post_Lvl_Ind', 'SOA_Antp_Lvl_TP', 'SOA_Guar_Lvl_TP']
    
    input_record = {f: sample[f] for f in features}
    
    # Prediction
    mortality_rate, shap_dict = predictor.predict(input_record)
    overall_rate = df['Death_Count'].sum() / df['Policies_Exposed'].sum()
    
    prediction = PredictionData(
        mortality_rate=mortality_rate,
        overall_death_rate=overall_rate,
        interpretation="higher" if mortality_rate > overall_rate else "lower"
    )
    
    # SHAP
    shap_contributions = []
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    for feat, shap_val in sorted_shap[:5]:
        direction = "Increases" if shap_val > 0 else "Decreases"
        shap_contributions.append(SHAPData(
            feature=feat, value=input_record.get(feat, 'N/A'),
            shap_value=shap_val, direction=direction
        ))
    
    # Population context
    percentile_context = []
    for feat in ['Attained_Age', 'Issue_Age', 'Duration']:
        val = sample[feat]
        pct = int((df[feat] < val).mean() * 100)
        percentile_context.append({
            "feature": feat, "value": val, "percentile": pct,
            "description": f"{feat}={val} is at the {pct}th percentile"
        })
    
    category_context = []
    for feat in ['Sex', 'Smoker_Status', 'Insurance_Plan']:
        val = sample[feat]
        pct = round((df[feat] == val).mean() * 100, 1)
        category_context.append({
            "feature": feat, "value": val, "population_pct": pct,
            "description": f"{feat}={val} ({pct}% of population)"
        })
    
    # Segment matching
    cov_seg = find_matching_segment(input_record, kb)
    coverage_segment = None
    if cov_seg:
        coverage_segment = SegmentData(
            segment_id=cov_seg.get('segment_id'),
            rule_text=cov_seg.get('rule_text'),
            ae_ratio=cov_seg.get('statistics', {}).get('ae_ratio', 1.0),
            credibility="high" if cov_seg.get('statistics', {}).get('exposure', 0) > 50000 else "medium"
        )
    
    # Spotlight: Try to match from knowledge base spotlight anomalies
    spotlight_segment = None
    spot_match = find_matching_spotlight(input_record, kb)
    if spot_match:
        spotlight_segment = SegmentData(
            segment_id=spot_match.get('segment_id', 'SPOT_UNKNOWN'),
            rule_text=' AND '.join(spot_match.get('rule_list', [])),
            ae_ratio=spot_match.get('ae_ratio', 1.0),
            credibility=str(spot_match.get('credibility', 'unknown')),
            risk_level="high_risk" if spot_match.get('ae_ratio', 1.0) > 1.5 else "elevated"
        )

    
    # Evidence charts
    evidence_charts = []
    charts_dir = Path(__file__).parent.parent.parent / 'data' / 'reports'
    if (charts_dir / 'shap_waterfall.png').exists():
        evidence_charts.append(EvidenceChart(
            name="SHAP Waterfall", path="shap_waterfall.png",
            description="Feature contributions to prediction"
        ))
    
    overall_ae = kb.overall_ae.get('overall_ae', 0.9989) if kb.overall_ae else 0.9989
    
    return ReportData(
        report_date=datetime.now().strftime('%Y-%m-%d'),
        report_version="1.0",
        input_record=input_record,
        prediction=prediction,
        shap_contributions=shap_contributions,
        percentile_context=percentile_context,
        category_context=category_context,
        coverage_segment=coverage_segment,
        spotlight_segment=spotlight_segment,
        overall_ae=overall_ae,
        evidence_charts=evidence_charts,
        disclaimers=[
            "This prediction is based on historical data from 2012-2019 and is for reference only.",
            "This model does not include COVID-19 period data; post-2020 predictions require adjustment.",
            "This report complies with ASOP 23/25/41/56 standards."
        ]
    )

def build_spotlight_data(kb, top_n: int = 5):
    """
    Build SpotlightReportData from spotlight segments.
    Rank by exposure * |A/E - 1| and take Top N.
    """
    exception_def = ExceptionDefinition(
        ae_deviation_threshold=0.15,
        min_exposure=1000,
        notes="Top anomalies ranked by exposure × |A/E−1|. Spotlight segments generated from semantic pipeline."
    )

    rows = []
    for seg in kb.spotlight_details.values():
        stats = seg.get("statistics", {})
        exposure = stats.get("exposure")
        ae = stats.get("ae_ratio")

        # skip if missing essentials
        if exposure is None or ae is None:
            continue

        score = float(exposure) * abs(float(ae) - 1.0)

        rows.append((score, seg, stats))

    # sort descending
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = rows[:top_n]

    anomalies = []
    for score, seg, stats in rows:
        anomalies.append(
            SpotlightAnomaly(
                segment_id=seg["segment_id"],
                rule_text=seg.get("rule_text", ""),
                ae_ratio=float(stats.get("ae_ratio")),
                exposure=float(stats.get("exposure")),
                # Backward compatible: older files used observed_deaths; newer files use actual_deaths
                deaths=stats.get("actual_deaths", stats.get("observed_deaths")),
                expected_deaths=stats.get("expected_deaths"),
                # Newer files provide a structured stability dict; render a compact note for prompt/report.
                stability_note=(
                    f"Year-based (col={stats.get('stability', {}).get('year_col')}): "
                    f"score={stats.get('stability', {}).get('stability_score')}, "
                    f"direction={stats.get('stability', {}).get('direction')}, "
                    f"anomaly_years={stats.get('stability', {}).get('anomaly_years')}/"
                    f"{stats.get('stability', {}).get('years_analyzed')}, "
                    f"persistence={stats.get('stability', {}).get('persistence')}, "
                    f"log_ae_sd={stats.get('stability', {}).get('log_ae_sd')}"
                    if isinstance(stats.get('stability'), dict) else
                    stats.get(
                        "stability_note",
                        "Not computed (year-level stability metrics not generated in spotlight details)."
                    )
                )
            )
        )

    return SpotlightReportData(
        report_date=str(date.today()),
        exception_definition=exception_def,
        top_anomalies=anomalies,
        disclaimers=[
            "Spotlight segments are intended for exception detection and hypothesis generation.",
            "Ranking uses exposure × |A/E−1| to prioritize segments with both material deviation and scale."
        ]
    )


def main():
    print("=" * 70)
    print("STRUCTURED ACTUARIAL REPORT GENERATOR")
    print("=" * 70)
    
    tracker = CostTracker()

    if REPORT_MODE == "spotlight":
        stage = tracker.start_stage("Spotlight Report")

        llm = create_llm_client()
        system_prompt = get_full_system_prompt()
        kb = get_knowledge_base()
        data = build_spotlight_data(kb)

        context = SpotlightPromptContext(
            exception_definition=data.exception_definition,
            top_anomalies=data.top_anomalies
        )

        insights, user_prompt = llm.generate_insights(
            context,
            system_prompt,
            output_schema=SpotlightInsights
        )

        report = FullReport(
            data=data,
            insights=insights,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            timing_summary=[],      # spotlight 不走 policy stages
            total_cost=0,
            total_tokens=0,
            is_validated=True
        )

        output_dir = Path(__file__).parent.parent.parent / 'data' / 'reports'
        renderer = create_renderer(str(output_dir))

        md_path = renderer.save_markdown(report, "spotlight_report.md")
        json_path = renderer.save_json(report, "spotlight_report.json")
        prompt_path = renderer.save_prompt(report, "spotlight_system_prompt_appendix.txt")

        tracker.end_stage(
            stage,
            input_text=system_prompt + "\n" + str(context.model_dump()),
            output_text=str(insights.model_dump())
        )

        print(f"\n[Spotlight] Report saved")
        print(f"    ✓ {md_path}")
        print(f"    ✓ {json_path}")
        print(f"    ✓ {prompt_path}")

        tracker.print_summary()

        print("\n" + "=" * 70)
        print("COMPLETE")
        print("=" * 70)
        return


    # Stage 1: Load data
    stage = tracker.start_stage("1. Load Data & Sample")
    sample, df = load_sample()
    if sample is None:
        return None
    tracker.end_stage(stage)
    print(f"\n[1] Data loaded ({stage['duration']:.2f}s)")
    print(f"    Records: {len(df):,}")
    print(f"    Sample: Age={sample['Attained_Age']}, Sex={sample['Sex']}, Smoker={sample['Smoker_Status']}")
    
    # Stage 2: Initialize
    stage = tracker.start_stage("2. Initialize Components")
    kb = get_knowledge_base()
    predictor = create_predictor()
    llm = create_llm_client()
    system_prompt = get_full_system_prompt()
    prompt_stats = get_prompt_stats(system_prompt)
    tracker.end_stage(stage)
    print(f"\n[2] Initialized ({stage['duration']:.2f}s)")
    print(f"    System prompt: {prompt_stats['characters']:,} chars (~{prompt_stats['estimated_tokens']:,} tokens)")
    
    # Stage 3: Create data
    stage = tracker.start_stage("3. Create Report Data")
    data = create_report_data(sample, df, predictor, kb)
    tracker.end_stage(stage)
    print(f"\n[3] Report data created ({stage['duration']:.2f}s)")
    print(f"    Mortality: {data.prediction.mortality_rate:.6f}")
    print(f"    A/E (Coverage): {data.coverage_segment.ae_ratio if data.coverage_segment else 'N/A'}")
    if data.coverage_segment:
        print(f"    Coverage: {data.coverage_segment.segment_id}")
    if data.spotlight_segment:
        print(f"    ⚠️ Spotlight: {data.spotlight_segment.segment_id}")
    
    # Stage 4: Build context
    stage = tracker.start_stage("4. Build LLM Context")
    context = PromptContext(
        prediction=data.prediction,
        shap_contributions=data.shap_contributions,
        percentiles=data.percentile_context,
        categories=data.category_context,
        coverage_segment=data.coverage_segment.model_dump() if data.coverage_segment else None,
        spotlight_segment=data.spotlight_segment.model_dump() if data.spotlight_segment else None,
        overall_ae=data.overall_ae
    )
    input_text = system_prompt + str(context.model_dump())
    tracker.end_stage(stage, input_text=input_text)
    print(f"\n[4] Context built ({stage['duration']:.3f}s)")
    
    # Stage 5: Generate insights
    stage = tracker.start_stage("5. Generate Insights (LLM)")
    insights, user_prompt = llm.generate_insights(context, system_prompt)
    full_prompt = system_prompt + "\n\n---\n\n" + user_prompt  # Full prompt for accurate token count
    output_text = str(insights.model_dump()) if hasattr(insights, 'model_dump') else str(insights.__dict__)
    tracker.end_stage(stage, input_text=full_prompt, output_text=output_text)
    print(f"\n[5] Insights generated ({stage['duration']:.2f}s)")
    print(f"    Cost: ${stage['cost']:.6f}")
    

    # Stage 6: Render
    stage = tracker.start_stage("6. Render Report")
    report = FullReport(
        data=data, 
        insights=insights,
        system_prompt=system_prompt, 
        user_prompt=user_prompt,
        timing_summary=tracker.stages.copy(),
        total_cost=sum(s['cost'] for s in tracker.stages),
        total_tokens=tracker.total_input_tokens + tracker.total_output_tokens,
        is_validated=True
    )

    
    output_dir = Path(__file__).parent.parent.parent / 'data' / 'reports'
    renderer = create_renderer(str(output_dir))
    
    md_path = renderer.save_markdown(report, "actuarial_report.md")
    json_path = renderer.save_json(report, "actuarial_report.json")
    prompt_path = renderer.save_prompt(report, "system_prompt_appendix.txt")
    tracker.end_stage(stage)
    print(f"\n[6] Report saved")
    print(f"    ✓ {md_path}")
    print(f"    ✓ {json_path}")
    print(f"    ✓ {prompt_path}")
    
    tracker.print_summary()
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return report


if __name__ == "__main__":
    main()
