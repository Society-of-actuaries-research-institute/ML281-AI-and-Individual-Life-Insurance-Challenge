# -*- coding: utf-8 -*-
"""
Prompt Builder

Builds complete system prompt with full knowledge base context.
"""

import json
import os
from pathlib import Path
from typing import Optional

from .config import KNOWLEDGE_BASE_DIR


def load_json(path: Path) -> dict:
    """Load JSON file safely."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"[WARN] Could not load {path}: {e}")
        return {}


def load_text(path: Path) -> str:
    """Load text file safely."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError) as e:
        print(f"[WARN] Could not load {path}: {e}")
        return ""


def build_system_prompt() -> str:
    """
    Build complete system prompt with all knowledge base context.

    Returns:
        Full system prompt string
    """
    kb_path = KNOWLEDGE_BASE_DIR
    sections = []

    # =========================================================================
    # 1. System Role + Task + Constraints
    # =========================================================================
    sections.append("""# System Role
You are an actuarial intelligence assistant supporting mortality experience analysis.
You are not the signing actuary. Your output is advisory and must be reviewed.

## Task
Generate actuarial narrative in strict JSON that matches the requested schema.
Use ONLY the evidence provided in the user prompt and the embedded knowledge base.
If evidence is missing or insufficient, state the limitation explicitly.

## Constraints
1. Model predicts **mortality rate** (expected deaths per exposure), NOT probability.
2. Use **associative language** (e.g., "associated with"), avoid causal claims.
3. Do not invent numbers, dates, or sources.
4. Follow ASOP 23/25/41/56 principles for data, credibility, modeling, and communication.
5. Output **valid JSON only**; no markdown, no commentary.
6. **FORBIDDEN**: Do not reference COVID-19, pandemic, or any years ≥ 2020. Data period is 2012-2019 only.
7. **FORBIDDEN**: Do not cite A/E ratios or mortality rates that differ from the exact values in the user prompt.
8. **REQUIRED**: All output must include the data limitation (2012-2019), ASOP compliance statement, and "for reference only" language.

## Style
- Language: English
- Tone: Professional actuarial memo
- Be concise, precise, and evidence-linked

## Numeric Formatting
- Mortality rate: 6 decimals (0.009666)
- A/E Ratio: 2-4 decimals (0.9989)
- SHAP value: signed, 4 decimals (+0.0352)
- Percentile: ordinal (90th percentile)
""")
    
    # =========================================================================
    # 2. Model Contract
    # =========================================================================
    contract = load_json(kb_path / 'contracts' / 'model_contract.json')
    if contract:
        limitations_data = contract.get("limitations", {})
        if isinstance(limitations_data, dict):
            limitations_list = [item for items in limitations_data.values() for item in items]
        else:
            limitations_list = limitations_data
            
        limitations_str = "\n".join(f"- {lim}" for lim in limitations_list)
        
        sections.append(f"""
---
## Model Specification (Contract)

**Model**: {contract.get('model_type', 'LightGBM')}
**Purpose**: {contract.get('purpose', 'Mortality prediction')}
**Training Period**: {contract.get('training_period', '2009-2019')}

### Limitations
{limitations_str}

### ASOP Compliance
{chr(10).join('- ' + c for c in contract.get('asop_compliance', []))}
""")
    
    # =========================================================================
    # 3. EDA Summary
    # =========================================================================
    numerical = load_json(kb_path / 'eda' / 'numerical_summary.json')
    categorical = load_json(kb_path / 'eda' / 'categorical_summary.json')
    
    if numerical:
        sections.append("""
--- 
## Data Distribution (EDA)

### Numerical Features""")
        for feat, stats in numerical.items():
            interp = stats.get('interpretation', {})
            sections.append(f"""
**{feat}**:
- Mean: {stats.get('mean', 'N/A')}, Std: {stats.get('std', 'N/A')}
- Range: [{stats.get('min', 0)}, {stats.get('max', 0)}]
- Low: {interp.get('low', 'N/A')}
- Typical: {interp.get('typical', 'N/A')}
- High: {interp.get('high', 'N/A')}
- Extreme: {interp.get('extreme', 'N/A')}""")
    
    if categorical:
        sections.append("""
### Categorical Features""")
        for feat, stats in categorical.items():
            if 'value_counts' in stats:
                top_values = list(stats['value_counts'].items())[:3]
                top_str = ', '.join([f"{v[0]}:{v[1]:.1%}" for v in top_values])
                sections.append(f"- **{feat}**: {top_str}")
    
    # =========================================================================
    # 4. Calibration Data
    # =========================================================================
    overall_ae = load_json(kb_path / 'calibration' / 'overall_ae.json')
    yearly_ae = load_json(kb_path / 'calibration' / 'yearly_ae.json')
    
    if overall_ae:
        sections.append(f"""
---
## Model Calibration

**Overall A/E Ratio**: {overall_ae.get('overall_ae', 0.9989):.4f}
- Total Actual Deaths: {overall_ae.get('total_actual_deaths', 0):,}
- Total Expected Deaths: {overall_ae.get('total_expected_deaths', 0):,.0f}
- Interpretation: {overall_ae.get('interpretation', 'Well-calibrated')}
- Data Period: {overall_ae.get('data_period', '2009-2019')}
""")
    
    if yearly_ae:
        sections.append("### Yearly A/E Trend")
        for year, ae in list(yearly_ae.items())[:5]:
            sections.append(f"- {year}: A/E = {ae:.4f}" if isinstance(ae, float) else f"- {year}: {ae}")
    
    # =========================================================================
    # 5. SHAP Feature Importance
    # =========================================================================
    global_shap = load_json(kb_path / 'shap' / 'global_importance.json')
    if global_shap:
        sections.append(f"""
---
## Feature Importance (SHAP)

{global_shap.get('description', 'Global SHAP feature importance')}
- **Base Value**: {global_shap.get('base_value', 'N/A')} ({global_shap.get('base_value_interpretation', 'N/A')})
- **Sample Size**: {global_shap.get('sample_size', 0):,}

### Global Ranking""")
        
        for item in global_shap.get('feature_importance', []):
            sections.append(
                f"- **{item.get('feature')}** (Rank {item.get('rank')}): "
                f"Mean |SHAP| = {item.get('mean_abs_shap', 0):.6f}, "
                f"Contribution = {item.get('pct_contribution', 0):.2f}%"
            )
    # =========================================================================
    # 6. EDA Insights (from insight.md if exists)
    # =========================================================================
    insight_text = load_text(kb_path / 'eda' / 'insight.md')
    if insight_text:
        sections.append(f"""
---
## Key EDA Insights

{insight_text}
""")

    # =========================================================================
    # 7. External Events / Domain Knowledge (optional)
    # =========================================================================
    external_text = load_text(kb_path / 'external' / 'context.md')
    events = load_json(kb_path / 'external' / 'events.json')

    if external_text or events:
        ext_section = ["---", "## External Context (Use as conditional context only)"]
        if external_text:
            ext_section.append(external_text)
        if events:
            ext_section.append("\n### Industry Event Timeline")
            ext_section.append("| Year | Type | Description |")
            ext_section.append("|------|------|-------------|")
            for ev in events:
                yr = ev.get("year_range") or ev.get("year", "N/A")
                ext_section.append(f"| {yr} | {ev.get('type', '')} | {ev.get('description', '')} |")
        sections.append("\n".join(ext_section))
    
    return "\n".join(sections)


def get_prompt_stats(prompt: str) -> dict:
    """Get statistics about the prompt."""
    lines = prompt.split('\n')
    chars = len(prompt)
    words = len(prompt.split())
    est_tokens = chars // 3  # Rough estimate
    
    return {
        'lines': len(lines),
        'characters': chars,
        'words': words,
        'estimated_tokens': est_tokens
    }


# Cache the built prompt — invalidated when knowledge_base/ directory mtime changes
_cached_prompt = None
_cached_prompt_mtime: Optional[float] = None


def _kb_mtime() -> Optional[float]:
    """Return the most recent mtime across all files in knowledge_base/."""
    try:
        kb_path = KNOWLEDGE_BASE_DIR
        mtimes = [p.stat().st_mtime for p in kb_path.rglob('*') if p.is_file()]
        return max(mtimes) if mtimes else None
    except Exception:
        return None


def get_full_system_prompt(force_rebuild: bool = False) -> str:
    """Get the full system prompt (cached, auto-invalidated on KB changes)."""
    global _cached_prompt, _cached_prompt_mtime
    current_mtime = _kb_mtime()
    if _cached_prompt is None or force_rebuild or (current_mtime != _cached_prompt_mtime):
        _cached_prompt = build_system_prompt()
        _cached_prompt_mtime = current_mtime
    return _cached_prompt
