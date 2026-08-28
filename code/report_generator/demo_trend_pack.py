# code/report_generator/generate_trend_pack.py
import os
import sys
import subprocess
from pathlib import Path
from .config import ILEC_WITH_PRED_PATH
from datetime import date
import json
import pandas as pd
from .schemas import FullReport, ExperienceTrendPackData, ExperienceTrendPackInsights
from .renderer import create_renderer  # 如果你是用 factory；否則改成 ReportRenderer(...)
from .llm import create_llm_client
from .prompt_builder import get_full_system_prompt

import subprocess
from pathlib import Path


def main():
    def build_charts_map(asset_names, base_dir="trend_pack/assets"):
        """
        Build a deterministic charts_map aligned to the report schema, based on known filenames.
        asset_names: list[str] of PNG filenames, e.g. ["age_curve.png", ...]
        returns: dict[str, list[str]] mapping section -> list of relative asset paths
        """
        # Normalize to relative paths used in prompts/reports
        rel_assets = [str(Path(base_dir) / n) for n in asset_names]

        # Helper: stable filter by exact filename (preferred)
        def by_name(name: str) -> list[str]:
            target = str(Path(base_dir) / name)
            return [p for p in rel_assets if p == target]

        # Helper: stable filter by substring (only when you intentionally want it)
        def by_substring(substr: str) -> list[str]:
            s = substr.lower()
            return [p for p in rel_assets if s in Path(p).name.lower()]

        charts_map = {
            # 0) Everything
            "all": rel_assets,

            # 1) Executive / Overview
            "overview": by_name("chart_1_overview.png") + by_name("chart_2_overview.png"),

            # 2) Core yearly comparison
            "yearly_compare": by_name("yearly_compare_year_sex.png"),

            # 3) Age curve diagnostics
            "age_curve": by_name("age_curve.png"),

            # 4) Portfolio / mix shift
            "mix_shift": by_name("product_mix_shift.png"),

            # 5) Product x year views (trend by product)
            "ae_by_product_year": by_name("ae_by_product_year.png"),
            "ae_by_product_year_age_split": by_name("ae_by_product_year_age_split.png"),

            # 6) Product x age x amount (segmented A/E)
            "ae_product_age_amount": by_name("ae_product_age_amount.png"),

            # 7) Issue year structure (deaths/policies)
            "issue_year_structure": (
                by_name("deaths_product_issue_year.png") +
                by_name("policies_product_issue_year.png")
            ),

            # 8) Placeholder for non-image evidence (e.g., drift flags table)
            "drift_flags": [],
        }

        # Optional: legacy catch-all (anything not already categorized)
        assigned = set(p for k, v in charts_map.items() if k != "all" for p in v)
        charts_map["legacy"] = [p for p in rel_assets if p not in assigned]
        full_rel_assets = [f"trend_pack/assets/{n}" for n in asset_names]

        return full_rel_assets, charts_map
    
    code_dir = Path(__file__).resolve().parents[1]  # .../code
    project_root = code_dir.parent                 # .../SOA_Challenge-main
    reports_dir = project_root / "data" / "reports"

    trend_pack_dir = reports_dir / "trend_pack"
    
    assets_dir = trend_pack_dir / "assets"
    tables_dir = trend_pack_dir / "tables"
    # assets_dir.mkdir(parents=True, exist_ok=True)
    # tables_dir.mkdir(parents=True, exist_ok=True)
    # print('code_dir', code_dir)
    # print('project_root', project_root)
    # print('reports_dir', reports_dir)
    # print('trend_pack_dir', trend_pack_dir)
    # 1) Run deterministic generators (charts + tables)
    env = dict(os.environ)
    env["TREND_PACK_OUTDIR"] = str(trend_pack_dir)
    env["ILEC_WITH_PRED_PATH"] = str(
        Path(os.environ.get("ILEC_WITH_PRED_PATH", ILEC_WITH_PRED_PATH))
    )

    gen_script = code_dir / "chart_generators" / "generate_charts_tested.py"
    subprocess.check_call([sys.executable, str(gen_script)], env=env)
    # print('gen_script', gen_script)

    # 2) Build deterministic report data (relative paths from data/reports/)
    
    # 2) Index deterministic artifacts (prefer manifest written by generator; fallback to scanning)
    manifest_path = trend_pack_dir / "artifacts_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        asset_names = manifest.get("assets", [])
        table_names = manifest.get("tables", [])
    else:
        asset_names = sorted([p.name for p in assets_dir.glob("*.png")])
        table_names = sorted([p.name for p in tables_dir.glob("*.csv")])

    # Tables: map stem -> relative path under data/reports/
    tables_map = {Path(n).stem: f"trend_pack/tables/{n}" for n in table_names}

    # Charts: group into sections expected by the report schema
    _, charts_map = build_charts_map(asset_names=asset_names)
    data = ExperienceTrendPackData(
        report_date=str(date.today()),
        definitions={
            "Exposure": "Sum of Policies_Exposed (policy-years).",
            "Observed Deaths": "Sum of Death_Count.",
            "Expected Deaths": "Sum of Expected_Deaths (model-based, consistent with A/E).",
            "A/E": "Observed Deaths / Expected Deaths.",
            "RR": "Group model_rate / overall model_rate (within the same year).",
            "Lift": "AE / Overall_AE (deterministic, prioritization-oriented).",
            "Drift flag": "|ΔAE| > 0.10 with Exposure >= 1000 for >=2 consecutive years (deterministic rule).",
        },
        tables=tables_map,
        charts=charts_map,
    )

    # 3) LLM narrative (optional; still returns deterministic MD/JSON even if empty)
    insights = ExperienceTrendPackInsights()
    llm = create_llm_client()
    system_prompt = get_full_system_prompt()

    user_prompt = ""
    if getattr(llm, "_initialized", False) and getattr(llm, "client", None):

        def _abs_from_reports(rel_path: str) -> Path:
            # rel_path is relative to data/reports/, e.g. "trend_pack/tables/x.csv"
            return reports_dir / rel_path

        def _csv_preview(rel_path: str, n_rows: int = 12) -> str:
            p = _abs_from_reports(rel_path)
            if not p.exists():
                return f"[Missing CSV at {rel_path}]"
            try:
                df = pd.read_csv(p)
            except Exception as e:
                return f"[Failed to read CSV {rel_path}: {e}]"
            head = df.head(n_rows)
            summary = f"shape={df.shape}, columns={list(df.columns)}"
            return summary + "\n" + head.to_markdown(index=False)

        def _png_meta(rel_path: str) -> str:
            p = _abs_from_reports(rel_path)
            if not p.exists():
                return f"[Missing PNG at {rel_path}]"
            size_kb = p.stat().st_size / 1024
            return f"file={rel_path} ({size_kb:.1f} KB)"

        # Build evidence blocks
        tables_lines = []
        for k, rel in sorted(data.tables.items()):
            tables_lines.append(f"- {k}: {rel}\n  " + _csv_preview(rel).replace("\n", "\n  "))

        charts_lines = []
        for section, rel_list in data.charts.items():
            if not rel_list:
                continue
            charts_lines.append(f"{section}:")
            for rel in rel_list:
                charts_lines.append(f"- " + _png_meta(rel))

        user_prompt = f"""
        You are writing an Experience Trend Pack for actuaries.
        Use associative language only; do not claim causality; do not invent numbers.
        You must only use the provided evidence blocks below (CSV previews + PNG file references).
        If evidence is missing or insufficient, state limitations explicitly.

        EVIDENCE: TABLES (CSV previews)
        {chr(10).join(tables_lines)}

        EVIDENCE: CHARTS (PNG files saved on disk)
        {chr(10).join(charts_lines)}

        Write JSON with keys:
        - executive_summary (5 bullets, concise)
        - yearly_compare_notes (bullets; reference tables/charts by name)
        - age_curve_notes (bullets)
        - mix_shift_notes (bullets)
        - drift_flags_notes (bullets; if drift_flags table missing, say so)
        - hypothesis_directions (5 bullets; hypotheses only, not conclusions)
        """
    full_prompt = f"system_prompt:{system_prompt}\n\n---\n\nuser_prompt:{user_prompt}"

    reviewed, _ = llm.generate_json_with_review(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        output_schema=ExperienceTrendPackInsights
    )

    if reviewed is not None and hasattr(reviewed, "model_dump"):
        insights = reviewed
    elif not getattr(llm, "_initialized", False) or not getattr(llm, "client", None):
        pass
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        txt = llm.client.invoke(messages).content
        parsed = None
        try:
            parsed = json.loads(txt)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            payload = parsed.get("report", parsed)
            insights.executive_summary = payload.get("executive_summary", "")
            insights.yearly_compare_notes = payload.get("yearly_compare_notes", "")
            insights.age_curve_notes = payload.get("age_curve_notes", "")
            insights.mix_shift_notes = payload.get("mix_shift_notes", "")
            insights.drift_flags_notes = payload.get("drift_flags_notes", "")
            insights.hypothesis_directions = payload.get("hypothesis_directions", []) or []
        else:
            insights.executive_summary = txt
    # 4) Render MD+JSON into data/reports/
    report = FullReport(
        data=data,
        insights=insights,
        system_prompt="",
        user_prompt="",
        timing_summary=[],
        total_cost=0,
        total_tokens=0,
        is_validated=True
    )

    renderer = create_renderer(str(reports_dir))
    
    md_path = renderer.save_markdown(report, "experience_trend_pack.md")
    json_path = renderer.save_json(report, "experience_trend_pack.json")
    
    # pdf_path = md_path.with_suffix(".pdf")
    # convert_md_to_pdf(md_path, pdf_path)
    print("✓", md_path)
    print("✓", json_path)
    # print("✓", pdf_path)

    prompt_path = trend_pack_dir / "full_prompt.txt"
    prompt_path.write_text(full_prompt, encoding="utf-8")
    print('prompt_path:', prompt_path)
if __name__ == "__main__":
    main()
