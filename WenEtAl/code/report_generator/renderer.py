# -*- coding: utf-8 -*-
"""
Report Renderer (English)

Renders structured report data to Markdown and PDF formats.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime
import json
import re
from .schemas import FullReport, ReportData, ActuarialInsights


class ReportRenderer:
    """
    Renders structured report to various output formats.
    
    Combines fixed data (tables, charts, disclaimers) with 
    LLM-generated insights into a complete actuarial report.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _render_spotlight_markdown(self, report):
        d = report.data
        i = report.insights
        out = []

        out.append("# Spotlight / Exception Report\n")
        out.append(f"**Report Date**: {d.report_date}\n\n---\n")

        out.append("## 1. Exception Definition\n")
        out.append(f"- Threshold: |A/E - 1| > {d.exception_definition.ae_deviation_threshold}\n")
        out.append(f"- Minimum Exposure: {d.exception_definition.min_exposure}\n\n---\n")

        out.append("## 2. Top Anomalies (Deterministic)\n")
        for a in d.top_anomalies:
            out.append(f"### {a.segment_id}\n")
            out.append(f"- Rule: {a.rule_text}\n")
            out.append(f"- **A/E Ratio**: {a.ae_ratio:.3f}  "
                       f"({'over-mortality' if a.ae_ratio > 1 else 'under-mortality'})\n")
            out.append(f"- Exposure (policy-years): {a.exposure:,.0f}\n")

            # Deaths & expected deaths
            if a.deaths is not None:
                out.append(f"- Observed Deaths: {a.deaths:,.0f}\n")
            else:
                out.append("- Observed Deaths: N/A\n")

            if getattr(a, "expected_deaths", None) is not None:
                out.append(f"- Expected Deaths: {a.expected_deaths:,.1f}\n")
            else:
                out.append("- Expected Deaths: N/A\n")

            # Financial impact (deaths delta)
            if getattr(a, "deaths_delta", None) is not None:
                direction_word = "excess" if a.deaths_delta > 0 else "deficit"
                out.append(
                    f"- **Net Deaths ({direction_word})**: {a.deaths_delta:+.1f} "
                    f"(actual minus expected)\n"
                )

            # Credibility
            if getattr(a, "credibility_score", None) is not None:
                cred_label = (
                    "High" if a.credibility_score >= 0.9
                    else "Medium" if a.credibility_score >= 0.5
                    else "Low"
                )
                out.append(
                    f"- Credibility Score: {a.credibility_score:.2f} ({cred_label})\n"
                )
            else:
                out.append("- Credibility Score: N/A\n")

            # Structured stability metrics
            if getattr(a, "stability_score", None) is not None:
                out.append(f"- Stability Score: {a.stability_score:.2f}  "
                           f"(fraction of years anomaly persists)\n")
            if getattr(a, "persistence", None) is not None:
                out.append(f"- Persistence: {a.persistence:.2f}\n")
            if getattr(a, "log_ae_sd", None) is not None:
                out.append(f"- Log A/E Std Dev: {a.log_ae_sd:.3f}  "
                           f"(volatility; lower = more stable)\n")

            stab = a.stability_note or "Stability metrics not computed at this segment level."
            out.append(f"- Stability Note: {stab}\n")
            out.append("\n")

        out.append("---\n## 3. Causal Hypothesis Assessment (LLM)\n")
        for item in i.items:
            out.append(f"### {item.segment_id}\n")
            out.append(f"{item.summary}\n\n")
            out.append(f"**Temporal stability**: {item.temporal_stability_interpretation}\n\n")

            q = item.why_might_happen
            out.append("**Why might this happen? (conditional, mechanism-based)**\n")
            out.append(f"- Data: {'; '.join(q.data_related)}\n")
            out.append(f"- Model: {'; '.join(q.model_related)}\n")
            out.append(f"- External: {'; '.join(q.external_factors)}\n")
            out.append(f"- Definition: {'; '.join(q.definition_segmentation)}\n\n")

            out.append("**Investigation path**\n")
            for step in item.investigation_path:
                out.append(f"- {step}\n")
            out.append("\n")

            out.append("**Recommended actions**\n")
            for act in item.recommended_actions:
                out.append(f"- {act}\n")
            out.append("\n---\n")

        out.append("## 4. Overall Recommended Actions\n")
        for act in i.overall_recommended_actions:
            out.append(f"- {act}\n")

        # LLM audit trail
        if report.llm_model_version:
            out.append(f"\n---\n*LLM model: {report.llm_model_version}*  \n")
        if report.total_tokens:
            out.append(f"*Tokens used: {report.total_tokens:,}*  \n")
        if report.validation_notes:
            out.append(f"*Validation notes: {'; '.join(report.validation_notes)}*\n")

        return "".join(out)

    def _maybe_json_block_to_markdown(self, text: str) -> str:
        """If `text` looks like a JSON block (optionally fenced), render it as readable Markdown.

        Minimal-change intent: keep existing behavior for non-JSON text.
        """
        if not text:
            return text

        s = text.strip()

        # Strip fenced code blocks like ```json ... ```
        if s.startswith("```"):
            lines = s.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                # drop first and last fence lines
                s = "\n".join(lines[1:-1]).strip()

        if not s or s[0] not in "{[":
            return text

        try:
            obj = json.loads(s)
        except Exception:
            return text

        return self._json_to_markdown(obj).rstrip() + "\n"

    def _json_to_markdown(self, obj, indent: int = 0) -> str:
        """Render python (dict/list/scalar) to a compact Markdown bullet structure."""
        pad = "  " * indent
        if isinstance(obj, dict):
            out = []
            for k, v in obj.items():
                key = str(k).replace("_", " ")
                if isinstance(v, (dict, list)):
                    out.append(f"{pad}- **{key}**\n")
                    out.append(self._json_to_markdown(v, indent + 1))
                else:
                    out.append(f"{pad}- **{key}**: {v}\n")
            return "".join(out)
        if isinstance(obj, list):
            out = []
            for v in obj:
                if isinstance(v, (dict, list)):
                    out.append(f"{pad}-\n")
                    out.append(self._json_to_markdown(v, indent + 1))
                else:
                    out.append(f"{pad}- {v}\n")
            return "".join(out)
        return f"{pad}- {obj}\n"
    
    
    def _render_trend_pack_markdown(self, report):
        """
        Render the Experience Trend Pack in Markdown.

        Assumptions (aligned to generate_charts_tested_with_csv.py outputs):
        - Every PNG saved under trend_pack/assets has a corresponding deterministic CSV
          with the same stem saved under trend_pack/tables.
            e.g., trend_pack/assets/age_curve.png -> trend_pack/tables/age_curve.csv
        - report.data.charts is a dict[str, list[str]] mapping section -> list of relative PNG paths.
        - report.data.tables may be present, but the renderer does not require it for per-figure tables
          because it can derive the deterministic CSV path from the PNG path.

        Notes:
        - The renderer embeds figures via Markdown image links. For deterministic evidence, it also
          prints the derived CSV path next to each figure.
        """
        from pathlib import Path

        def _png_name(p: str) -> str:
            return Path(p).name

        def _fig_title(p: str) -> str:
            # Title mapping for known figures; fallback to filename.
            name = _png_name(p)
            mapping = {
                # Overview
                "chart_1_overview.png": "Overview: A/E by Observation Year (Count) with Death Volume",
                "chart_2_overview.png": "Overview: A/E by Face Amount Band and Observation Year",

                # Core Trend Pack
                "yearly_compare_year_sex.png": "A/E by Observation Year (by Sex)",
                "age_curve.png": "Age Curve (Empirical vs Model)",
                "product_mix_shift.png": "Product Mix Shift (Exposure Share by Year)",

                # Product / segmentation extensions
                "ae_by_product_year.png": "A/E by Product and Observation Year",
                "ae_by_product_year_age_split.png": "A/E by Product and Observation Year (Age Band Split)",
                "ae_product_age_amount.png": "A/E by Product (Age Band × Face Amount Band)",

                # Issue year structure
                "deaths_product_issue_year.png": "Deaths by Product and Issue Year",
                "policies_product_issue_year.png": "Policies Exposed by Product and Issue Year",
            }
            return mapping.get(name, name)

        def _csv_for_png(png_rel: str) -> str:
            # Derive deterministic CSV path from PNG path
            # trend_pack/assets/X.png -> trend_pack/tables/X.csv
            if "/assets/" in png_rel:
                return png_rel.replace("/assets/", "/tables/").replace(".png", ".csv")
            # fallback: same folder
            return str(Path(png_rel).with_suffix(".csv"))

        d = report.data
        i = getattr(report, "insights", None)

        # Helper to render a section of charts with per-figure deterministic table references.
        def _render_chart_section(section_title: str, section_key: str, commentary_attr: str | None = None, start_fig_no: int = 1):
            nonlocal out
            fig_no = start_fig_no

            rel_list = []
            if getattr(d, "charts", None):
                rel_list = d.charts.get(section_key, []) or []

            if not rel_list:
                return fig_no  # no change

            out.append(f"## {section_title}\n\n")

            for p in rel_list:
                title = _fig_title(p)
                csv_path = _csv_for_png(p)

                # out.append(f"**{fig_no}. {title}**\n\n")
                # out.append(f"Deterministic table: `{csv_path}`\n\n")
                out.append(f"![{title}]({p})\n\n")

                # Section-level commentary (LLM) if present
                if commentary_attr and i and getattr(i, commentary_attr, "").strip():
                    out.append("**Commentary (LLM)**\n\n")
                    out.append(getattr(i, commentary_attr).strip() + "\n\n")

                fig_no += 1

            out.append("---\n\n")
            return fig_no

        out = []
        out.append("# Experience Trend Pack (Deterministic Evidence + LLM Commentary)\n\n")
        
        def _extract_json_from_codeblock(s: str) -> dict | None:
            """
            If s looks like ```json ... ``` (or ``` ... ```), extract and parse JSON.
            Return dict if successful, else None.
            """
            if not isinstance(s, str):
                return None

            # Match ```json ... ``` or ``` ... ```
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
            if not m:
                return None

            payload = m.group(1).strip()
            try:
                obj = json.loads(payload)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None


        # --- inside your renderer, replace your current if i: block with this ---
        if i:
            insights_dict = i.model_dump() if hasattr(i, "model_dump") else i.dict()

            # 1) If the model stuffed the full JSON into executive_summary as a code block, parse it
            parsed = _extract_json_from_codeblock(insights_dict.get("executive_summary"))
            if parsed:
                insights_dict = parsed  # overwrite with the real structured JSON

            # 2) Render in a stable order (optional but recommended)
            preferred_order = [
                "executive_summary",
                "yearly_compare_notes",
                "age_curve_notes",
                "mix_shift_notes",
                "drift_flags_notes",
                "hypothesis_directions",
            ]
            ordered_items = [(k, insights_dict.get(k)) for k in preferred_order if k in insights_dict]
            # include any extra keys not in preferred_order
            ordered_items += [(k, v) for k, v in insights_dict.items() if k not in dict(ordered_items)]
            insight_lines = []
            for k, v in ordered_items:
                # Skip empty sections (your current dict has many empty strings)
                if v is None:
                    continue
                if isinstance(v, str) and not v.strip():
                    continue
                if isinstance(v, (list, tuple)) and len(v) == 0:
                    continue

                insight_lines.append(f"## {k}\n")

                if isinstance(v, (list, tuple)):
                    for b in v:
                        bb = str(b).strip()
                        if bb:
                            insight_lines.append(f"- {bb}\n")
                    insight_lines.append("\n")
                else:
                    insight_lines.append(str(v).strip() + "\n\n")

            out.append("".join(insight_lines) + "\n---\n\n")

        # 0) Overview
        fig_no = 1
        fig_no = _render_chart_section("0. Overview", "overview", None, fig_no)

        # 1) Yearly compare (Year × Sex)
        fig_no = _render_chart_section("1. Yearly Compare (Year × Sex)", "yearly_compare", "yearly_compare_notes", fig_no)

        # 2) Age curve
        fig_no = _render_chart_section("2. Age Curve (Empirical vs Model)", "age_curve", "age_curve_notes", fig_no)

        # 3) Product mix shift
        fig_no = _render_chart_section("3. Product Mix Shift", "mix_shift", "mix_shift_notes", fig_no)

        # 4) A/E by product and year
        fig_no = _render_chart_section("4. A/E by Product and Observation Year", "ae_by_product_year", None, fig_no)

        # 5) A/E by product and year (age split)
        fig_no = _render_chart_section("5. A/E by Product and Observation Year (Age Band Split)", "ae_by_product_year_age_split", None, fig_no)

        # 6) A/E by product (age × face amount)
        fig_no = _render_chart_section("6. A/E by Product (Age Band × Face Amount Band)", "ae_product_age_amount", None, fig_no)

        # 7) Issue year structure
        fig_no = _render_chart_section("7. Issue Year Structure (Deaths / Policies)", "issue_year_structure", None, fig_no)

        # 8) Drift flags (table-driven; chart optional)
        out.append("## 8. Drift Flags (Structural Break)\n\n")
        drift_table = ""
        if getattr(d, "tables", None):
            drift_table = d.tables.get("drift_flags", "") or ""
        # If you follow the "png->csv" rule, drift_flags may exist only as CSV (no PNG).
        if drift_table:
            out.append(f"Deterministic table: `{drift_table}`\n\n")
        else:
            # fallback: common default path used by the generator
            out.append("Deterministic table: `trend_pack/tables/drift_flags.csv` (if generated)\n\n")

        # Optional drift flag charts
        rel_list = []
        if getattr(d, "charts", None):
            rel_list = d.charts.get("drift_flags", []) or []
        if rel_list:
            local_no = 1
            for p in rel_list:
                title = _fig_title(p)
                out.append(f"**{local_no}. {title}**\n\n")
                out.append(f"![{local_no}. {title}]({p})\n\n")
                local_no += 1

        if i and getattr(i, "drift_flags_notes", "").strip():
            out.append("**Interpretation (LLM, non-deterministic)**\n\n")
            out.append(i.drift_flags_notes.strip() + "\n\n")

        # Hypotheses
        if i and getattr(i, "hypothesis_directions", None):
            out.append("## 9. Hypothesis Directions (LLM)\n\n")
            if isinstance(i.hypothesis_directions, (list, tuple)):
                for k, h in enumerate(i.hypothesis_directions, 1):
                    out.append(f"{k}. {str(h).strip()}\n")
            else:
                out.append(str(i.hypothesis_directions).strip() + "\n")
            out.append("\n")

        # Definitions appendix (deterministic)
        if getattr(d, "definitions", None):
            out.append("\n---\n\n## Appendix: Metric Definitions (Deterministic)\n\n")
            for k, v in d.definitions.items():
                out.append(f"- **{k}**: {v}\n")

        return "".join(out)

    def _render_validation_markdown(self, report):
        d = report.data
        out = []

        out.append("# Validation Diagnostic Report\n")
        out.append(f"**Report Date**: {d.report_date}\n\n---\n")

        out.append("## 1. Validation Checks\n")
        for c in d.validation_checks:
            out.append(f"- **{c.check_id}** [{c.status.upper()}] {c.description}\n")
            if c.evidence:
                out.append(f"  - Evidence: {c.evidence}\n")

        out.append("\n---\n## 2. Calibration Summary (Deterministic)\n")
        overall = d.calibration_overall or {}
        if overall:
            out.append(f"- Overall A/E: {overall.get('overall_ae', 'N/A')}\n")
            out.append(f"- Total Actual Deaths: {overall.get('total_actual_deaths', 'N/A')}\n")
            out.append(f"- Total Expected Deaths: {overall.get('total_expected_deaths', 'N/A')}\n")
            out.append(f"- Period: {overall.get('data_period', 'N/A')}\n")
        else:
            out.append("- Overall calibration data not available\n")

        out.append("\n## 3. Yearly A/E Series (Deterministic)\n")
        if d.calibration_yearly:
            for year, ae in d.calibration_yearly.items():
                out.append(f"- {year}: {ae}\n")
        else:
            out.append("- Yearly calibration data not available\n")

        out.append("\n---\n## 4. Data Quality Flags (Deterministic)\n")
        if d.data_quality:
            for k, v in d.data_quality.items():
                out.append(f"- {k}: {v}\n")
        else:
            out.append("- Data quality flags not available\n")

        out.append("\n---\n## 5. Model Contract (Deterministic)\n")
        contract = d.model_contract or {}
        if contract:
            out.append(f"- Model: {contract.get('model_name', 'N/A')}\n")
            purpose = contract.get("purpose", {})
            if isinstance(purpose, dict):
                out.append(f"- Intended use: {purpose.get('intended_use', 'N/A')}\n")
                out.append(f"- Scope: {purpose.get('scope', 'N/A')}\n")
            limitations = contract.get("limitations", {})
            if isinstance(limitations, dict):
                out.append("- Limitations:\n")
                for group, items in limitations.items():
                    if not isinstance(items, list):
                        continue
                    for item in items:
                        out.append(f"  - {group}: {item}\n")
        else:
            out.append("- Model contract not available\n")

        # Drift flags (deterministic, from _compute_drift_flags)
        drift_flags = getattr(d, "drift_flags", None)
        if drift_flags:
            out.append("\n---\n## 6. Drift Flags (Deterministic)\n")
            out.append(
                "> **Flag rule**: |A/E − 1| > 0.10 persisting ≥ 2 consecutive years "
                "(primary); or year-over-year |ΔA/E| > 0.05 (sudden shift).\n\n"
            )
            out.append("| Year | A/E | ΔA/E | Description |\n")
            out.append("|------|-----|------|-------------|\n")
            for f in drift_flags:
                delta_str = f"{f.delta_ae:+.4f}" if f.delta_ae is not None else "—"
                out.append(
                    f"| {f.year} | {f.ae_ratio:.4f} | {delta_str} "
                    f"| {f.description} |\n"
                )
            out.append("\n")
        else:
            out.append("\n---\n## 6. Drift Flags (Deterministic)\n")
            out.append("- No drift flags detected (all years within |A/E-1| <= 0.10 or "
                       "no persistent consecutive deviation).\n")

        if getattr(d, "disclaimers", None):
            out.append("\n---\n## 7. Disclaimers\n")
            for item in d.disclaimers:
                out.append(f"- {item}\n")

        # LLM audit trail (transparency)
        if report.llm_model_version:
            out.append(f"\n---\n*LLM model: {report.llm_model_version}*  \n")
        if report.total_tokens:
            out.append(f"*Tokens used: {report.total_tokens:,}*  \n")
        if report.validation_notes:
            out.append(f"*Validation notes: {'; '.join(report.validation_notes)}*\n")

        return "".join(out)

    def render_markdown(self, report: FullReport) -> str:
        """Render complete report as Markdown."""
        data = report.data
        insights = report.insights
        
        sections = []

        if hasattr(report.data, "report_type") and report.data.report_type == "validation_diagnostic":
            return self._render_validation_markdown(report)

        if hasattr(report.data, "report_type") and report.data.report_type == "trend_pack":
            return self._render_trend_pack_markdown(report)
        
        if hasattr(report.data, "top_anomalies"):
            return self._render_spotlight_markdown(report)
        # =====================================================================
        # Header
        # =====================================================================
        sections.append(f"""# Mortality Prediction Analysis Report

**Report Date**: {data.report_date}  
**Version**: {data.report_version}  

---
""")
        
        # =====================================================================
        # 1. Executive Summary (LLM-generated)
        # =====================================================================
        sections.append(f"""## 1. Executive Summary（LLM）

{insights.summary}

---
""")
        
        # =====================================================================
        # 2. Prediction Results (Fixed Data)
        # =====================================================================
        sections.append(f"""## 2. Prediction Results

| Metric | Value | Description |
|:-------|:------|:------------|
| Predicted Mortality Rate | {data.prediction.mortality_rate:.6f} | Expected deaths per exposure-year |
| Overall Average Rate | {data.prediction.overall_death_rate:.6f} | Reference baseline |

**Risk Interpretation（LLM）**: {insights.risk_interpretation}

---
""")
        
        # =====================================================================
        # 3. SHAP Analysis (Fixed table + LLM explanations)
        # =====================================================================
        shap_table = "| Feature | Value | SHAP Contribution | Direction |\n|:--------|:------|:------------------|:----------|\n"
        for s in data.shap_contributions[:5]:
            sign = "+" if s.shap_value > 0 else ""
            shap_table += f"| {s.feature} | {s.value} | {sign}{s.shap_value:.4f} \n"
        
        shap_explanations = ""
        for exp in insights.shap_explanations:
            shap_explanations += f"- **{exp.feature}** ({exp.value}): {exp.explanation}\n"
        
        sections.append(f"""## 3. Risk Factor Analysis (SHAP)

### 3.1 Feature Contribution Table
{shap_table}

### 3.2 Feature Explanations
{shap_explanations}

---
""")
        
        # =====================================================================
        # 4. Population Context (Fixed Data)
        # =====================================================================
        percentile_text = ""
        for p in data.percentile_context:
            percentile_text += f"- {p.get('description', p.get('feature'))}\n"
        
        category_text = ""
        for c in data.category_context:
            category_text += f"- {c.get('description', c.get('feature'))}\n"
        
        sections.append(f"""## 4. Population Context

### 4.1 Numerical Feature Percentiles
{percentile_text}

### 4.2 Categorical Feature Distribution
{category_text}

---
""")
        
        # =====================================================================
        # 5. Segment Analysis (Fixed + LLM insight)
        # =====================================================================
        segment_section = "## 5. Segment Analysis\n\n"
        
        if data.coverage_segment:
            seg = data.coverage_segment
            segment_section += f"""### 5.1 Coverage Segment (General Classification)

This policy belongs to the following risk segment based on key characteristics:

| Metric | Value |
|:-------|:------|
| Segment ID | {seg.segment_id} |
| Classification Rule | {seg.rule_text} |
| A/E Ratio | {seg.ae_ratio:.3f} |
| Credibility | {seg.credibility} |

"""
        
        if data.spotlight_segment:
            seg = data.spotlight_segment
            segment_section += f"""### 5.2 ⚠️ Spotlight Alert (Anomaly Detection)

> **Note**: Spotlight segments identify statistically anomalous risk groups. This is an **additional warning**, not a contradiction to the Coverage segment.

| Metric | Value |
|:-------|:------|
| Segment ID | {seg.segment_id} |
| Classification Rule | {seg.rule_text} |
| Risk Level | {seg.risk_level} |
| A/E Ratio | {seg.ae_ratio:.2f} |

"""
        
        if insights.segment_insight:
            segment_section += f"**Segment Interpretation**: {insights.segment_insight}\n"
        
        segment_section += "\n---\n"
        sections.append(segment_section)
        
        # =====================================================================
        # 6. Model Calibration (Fixed Data + LLM note)
        # =====================================================================
        sections.append(f"""## 6. Model Calibration

| Metric | Value |
|:-------|:------|
| Overall A/E Ratio | {data.overall_ae:.4f} |

{insights.calibration_note}

---
""")
        
        # =====================================================================
        # 7. Evidence Charts (Fixed paths)
        # =====================================================================
        if data.evidence_charts:
            chart_section = "## 7. Evidence Charts\n\n"
            for chart in data.evidence_charts:
                chart_section += f"### {chart.name}\n"
                chart_section += f"![{chart.name}]({chart.path})\n\n"
                chart_section += f"*{chart.description}*\n\n"
            chart_section += "---\n"
            sections.append(chart_section)
        
        # =====================================================================
        # 8. Recommendations (LLM-generated)
        # =====================================================================
        if insights.recommendations:
            rec_section = "## 8. Recommendations\n\n"
            for i, rec in enumerate(insights.recommendations, 1):
                rec_section += f"{i}. {rec}\n"
            rec_section += "\n---\n"
            sections.append(rec_section)
        
        # =====================================================================
        # 9. Disclaimers (Fixed)
        # =====================================================================
        disclaimer_section = "## 9. Disclaimers\n\n"
        for disc in data.disclaimers:
            disclaimer_section += f"> {disc}\n\n"
        sections.append(disclaimer_section)
        

        
        # =====================================================================
        # Footer
        # =====================================================================
        sections.append(f"""
---

*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Compliant with ASOP 23/25/41/56 standards*
""")

        
        return "\n".join(sections)
    
    def save_prompt(self, report: FullReport, filename: str) -> str:
        """Save the system prompt, user prompt, and timing data to a text file."""
        if not report.system_prompt:
            return ""
        
        path = Path(self.output_dir) / filename
        with open(path, 'w', encoding='utf-8') as f:
            # Section 1: System Prompt
            f.write("=" * 70 + "\n")
            f.write("SECTION 1: SYSTEM PROMPT\n")
            f.write("=" * 70 + "\n\n")
            f.write(report.system_prompt)
            f.write("\n\n")
            
            # Section 2: User Prompt
            if report.user_prompt:
                f.write("=" * 70 + "\n")
                f.write("SECTION 2: USER PROMPT (sent to LLM)\n")
                f.write("=" * 70 + "\n\n")
                f.write(report.user_prompt)
                f.write("\n\n")
            
            # Section 3: Timing & Cost Summary
            if report.timing_summary:
                f.write("=" * 70 + "\n")
                f.write("SECTION 3: TIMING & COST SUMMARY\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"{'Stage':<35} {'Time':>10} {'Tokens':>10} {'Cost':>12}\n")
                f.write("-" * 70 + "\n")
                for stage in report.timing_summary:
                    tokens = stage.get('input_tokens', 0) + stage.get('output_tokens', 0)
                    f.write(f"{stage.get('name', 'Unknown'):<35} {stage.get('duration', 0):>9.2f}s {tokens:>9,} ${stage.get('cost', 0):>10.6f}\n")
                f.write("-" * 70 + "\n")
                total_tokens = report.total_tokens or 0
                total_cost = report.total_cost or 0
                f.write(f"{'TOTAL':<35} {'-':>10} {total_tokens:>9,} ${total_cost:>10.6f}\n")
        
        return str(path)
    
    def save_markdown(self, report: FullReport, filename: str = "report.md") -> Path:
        """Save report as Markdown file."""
        md_content = self.render_markdown(report)
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        return output_path
    
    def render_pdf(self, report: FullReport, filename: str = "report.pdf") -> Optional[Path]:
        """
        Render report as PDF.
        
        Requires weasyprint: pip install weasyprint
        """
        try:
            from weasyprint import HTML, CSS
        except ImportError:
            print("⚠ weasyprint not installed. Install with: pip install weasyprint")
            return None
        
        # Convert Markdown to HTML
        md_content = self.render_markdown(report)
        
        try:
            import markdown
            html_body = markdown.markdown(
                md_content, 
                extensions=['tables', 'fenced_code']
            )
        except ImportError:
            # Fallback: basic conversion
            html_body = f"<pre>{md_content}</pre>"
        
        # Wrap in HTML template
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #2980b9; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        blockquote {{ 
            background: #f9f9f9; 
            border-left: 4px solid #e74c3c; 
            padding: 10px 20px; 
            margin: 10px 0;
        }}
        img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
        
        output_path = self.output_dir / filename
        HTML(string=html_content).write_pdf(output_path)
        return output_path
    
    def save_json(self, report: FullReport, filename: str = "report.json") -> Path:
        """Save report as JSON for API consumption."""
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report.model_dump(), f, indent=2, ensure_ascii=False, default=str)
        return output_path


def create_renderer(output_dir: Optional[str] = None) -> ReportRenderer:
    """Factory function for ReportRenderer."""
    return ReportRenderer(output_dir)
