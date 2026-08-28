# -*- coding: utf-8 -*-
"""
LLM Module with Structured Output

Handles LLM integration for report generation using Google Gemini (primary)
or Anthropic Claude (fallback), with Pydantic structured output, retry logic,
timeout enforcement, and two-pass self-review.
"""

import os
import json
import concurrent.futures
from dataclasses import dataclass
from typing import Optional, Type, TypeVar
from pathlib import Path
from .schemas import (
    ActuarialInsights,
    SpotlightInsights,
    SpotlightPromptContext
)

# Load environment variables from .env (optional)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

# Check for pydantic schemas
try:
    from .schemas import ActuarialInsights, PromptContext, SHAPExplanation, HAS_PYDANTIC
except ImportError:
    HAS_PYDANTIC = False
    ActuarialInsights = None
    PromptContext = None
    SHAPExplanation = None

# Check for langchain-google-genai
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# Check for tenacity retry
try:
    from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False

from .config import LLM_TIMEOUT_SECONDS, LLM_MAX_RETRIES, LLM_BACKEND

# Type variable for structured output
T = TypeVar('T')


class LLMUnavailableError(RuntimeError):
    """Raised when LLM is not initialized and a real response is required."""


@dataclass
class UsageRecord:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost_usd(self, backend: str = "gemini") -> float:
        """Estimate cost in USD. Gemini Flash: $0.075/1M input, $0.30/1M output."""
        if backend == "gemini":
            return (self.input_tokens * 0.075 + self.output_tokens * 0.30) / 1_000_000
        # Claude Sonnet 4.6: $3/1M input, $15/1M output
        return (self.input_tokens * 3.0 + self.output_tokens * 15.0) / 1_000_000


class LLMClient:
    """
    LLM client for generating structured actuarial insights.

    Primary backend: Google Gemini (gemini-2.5-flash-lite).
    Fallback backend: Anthropic Claude Sonnet 4.6 (set LLM_BACKEND=claude).
    Features: tenacity retry, timeout enforcement, two-pass self-review,
              token/cost tracking.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-lite",
        temperature: float = 0.3,
        self_review: bool = True,
        backend: str = LLM_BACKEND,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.self_review = self_review
        self.backend = backend
        self.client = None
        self._review_client = None  # temperature=0 variant for deterministic self-review
        self._initialized = False
        self.usage = UsageRecord()

    @property
    def model_version(self) -> str:
        """Return full model identifier for audit trail."""
        if not self._initialized:
            return "not_initialized"
        if self.backend == "claude":
            return "claude-sonnet-4-6 (anthropic)"
        return f"{self.model_name} (google)"

    def is_available(self) -> bool:
        """Return True if LLM is initialized and ready."""
        return self._initialized and self.client is not None

    def initialize(self) -> 'LLMClient':
        """Initialize the LLM client based on configured backend."""
        if self._initialized:
            return self

        if self.backend == "claude":
            self._init_claude()
        else:
            self._init_gemini()

        return self

    def _init_gemini(self):
        if not HAS_GOOGLE:
            print("[WARN] langchain-google-genai not installed")
            print("  Install with: pip install langchain-google-genai")
            return

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[WARN] GOOGLE_API_KEY not found in environment")
            return

        try:
            self.client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=api_key
            )
            # Separate zero-temperature client for deterministic self-review pass
            self._review_client = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0,
                google_api_key=api_key
            )
            self._initialized = True
            print(f"[OK] LLM initialized: {self.model_name} (gemini)")
        except Exception as e:
            print(f"[WARN] Gemini initialization failed: {e}")

    def _init_claude(self):
        try:
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                print("[WARN] ANTHROPIC_API_KEY not found in environment")
                return
            self._claude_client = Anthropic(api_key=api_key)
            self._initialized = True
            print(f"[OK] LLM initialized: claude-sonnet-4-6 (claude)")
        except ImportError:
            print("[WARN] anthropic package not installed --install with: pip install anthropic")
        except Exception as e:
            print(f"[WARN] Claude initialization failed: {e}")

    def generate_insights(
        self,
        context,
        system_prompt: str,
        output_schema=None,
    ):
        """
        Generate structured actuarial insights.

        Raises LLMUnavailableError if not initialized.
        Returns (insights_object, user_prompt_str).
        """
        if output_schema is None:
            output_schema = ActuarialInsights

        if not self.is_available():
            raise LLMUnavailableError(
                "LLM is not initialized. Check GOOGLE_API_KEY or ANTHROPIC_API_KEY "
                "environment variables and ensure the required packages are installed."
            )

        # Build user prompt from context
        if isinstance(context, SpotlightPromptContext):
            user_prompt = self._build_spotlight_context_prompt(context)
        else:
            user_prompt = self._build_context_prompt(context)

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        try:
            response = self._invoke_with_timeout(full_prompt, output_schema)
            if self.self_review:
                response = self._self_review(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    draft=response,
                    output_schema=output_schema
                )
            return response, user_prompt
        except LLMUnavailableError:
            raise
        except Exception as e:
            print(f"[WARN] Structured output failed: {e}")
            if output_schema == SpotlightInsights:
                raise RuntimeError(
                    "Spotlight report requires SpotlightInsights structured output; "
                    "fallback parsing is disabled to prevent schema mismatch."
                )
            print("  Falling back to JSON parsing...")
            draft = self._fallback_generate(full_prompt)
            if self.self_review:
                draft = self._self_review(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    draft=draft,
                    output_schema=output_schema
                )
            return draft, user_prompt

    def _invoke_with_timeout(self, full_prompt: str, output_schema, client=None):
        """Call LLM with timeout + tenacity retry. Uses self.client unless client is provided."""
        active_client = client if client is not None else self.client

        def _call():
            structured_llm = active_client.with_structured_output(output_schema)
            result = structured_llm.invoke(full_prompt)
            # Track usage if metadata available
            if hasattr(result, 'usage_metadata') and result.usage_metadata:
                meta = result.usage_metadata
                self.usage.input_tokens += getattr(meta, 'input_tokens', 0)
                self.usage.output_tokens += getattr(meta, 'output_tokens', 0)
            return result

        if HAS_TENACITY:
            @retry(
                wait=wait_exponential(multiplier=1, min=2, max=30),
                stop=stop_after_attempt(LLM_MAX_RETRIES),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            )
            def _call_with_retry():
                return _call()
            target = _call_with_retry
        else:
            target = _call

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(target)
            try:
                return future.result(timeout=LLM_TIMEOUT_SECONDS)
            except concurrent.futures.TimeoutError:
                raise TimeoutError(
                    f"LLM call timed out after {LLM_TIMEOUT_SECONDS}s"
                )

    def generate_json_with_review(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema
    ):
        """Generate structured JSON (pydantic schema) with optional self-review."""
        if not self.is_available():
            return None, user_prompt

        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = self._invoke_with_timeout(full_prompt, output_schema)

        if self.self_review:
            response = self._self_review(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                draft=response,
                output_schema=output_schema
            )

        return response, user_prompt

    def _build_spotlight_context_prompt(self, context: SpotlightPromptContext) -> str:
        lines = []
        lines.append("# Spotlight / Exception Report")
        lines.append(
            "When discussing potential causes, use conditional and mechanism-based language. "
            "Each explanation must state assumptions, describe mechanisms, and indicate how "
            "the hypothesis could be empirically evaluated. Avoid definitive causal claims."
        )

        ed = context.exception_definition
        lines.append("\n## Exception Definition")
        lines.append(f"- |A/E - 1| > {ed.ae_deviation_threshold}")
        lines.append(f"- Minimum exposure: {ed.min_exposure}")
        if ed.notes:
            lines.append(f"- Notes: {ed.notes}")

        lines.append("\n## Top Anomalies (Deterministic)")
        for a in context.top_anomalies:
            lines.append(f"\n### {a.segment_id}")
            lines.append(f"- Rule: {a.rule_text}")
            lines.append(f"- A/E: {a.ae_ratio:.3f}")
            lines.append(f"- Exposure: {a.exposure:,.0f}")

            if a.deaths is None:
                lines.append("- Observed deaths: N/A (not available at segment level)")
            else:
                lines.append(f"- Observed deaths: {a.deaths:,.0f}")

            if getattr(a, "expected_deaths", None) is None:
                lines.append("- Expected deaths: N/A (not available at segment level)")
            else:
                lines.append(f"- Expected deaths: {a.expected_deaths:,.1f}")

            if a.years_present:
                lines.append(f"- Years present: {a.years_present}")
            if a.concentrated_years:
                lines.append(f"- Concentrated years: {a.concentrated_years}")

            if a.stability_note:
                lines.append(f"- Stability: {a.stability_note}")
            else:
                lines.append(
                    "- Stability: Not computed "
                    "(year-level A/E stability metrics were not generated for spotlight segments)"
                )

        lines.append(
            "\nOutput must be valid JSON matching SpotlightInsights schema."
        )
        lines.append(
            "\nREMINDER — HARD CONSTRAINTS (violation will be flagged by automated checker):\n"
            "- Data period is 2012-2019 ONLY. Do NOT reference COVID-19, pandemic, or any year >= 2020.\n"
            "- Use ONLY A/E values shown above. Do not introduce other numeric estimates.\n"
            "- All required disclaimers (data limitation, ASOP, reference-only) MUST appear in output."
        )
        return "\n".join(lines)

    def _build_context_prompt(self, context) -> str:
        """Build user prompt from context data."""
        lines = []

        lines.append("## Prediction Data")
        lines.append(f"- Predicted mortality rate: {context.prediction.mortality_rate:.6f}")
        lines.append(f"- Overall average rate: {context.prediction.overall_death_rate:.6f}")

        lines.append("\n## SHAP Contributions")
        for s in context.shap_contributions[:5]:
            sign = "+" if s.shap_value > 0 else ""
            lines.append(f"- {s.feature}: {s.value} -> SHAP={sign}{s.shap_value:.4f} ({s.direction})")

        lines.append("\n## Population Context")
        for p in context.percentiles:
            lines.append(f"- {p.get('description', str(p))}")
        for c in context.categories:
            lines.append(f"- {c.get('description', str(c))}")

        if context.coverage_segment:
            seg = context.coverage_segment
            lines.append(f"\n## Coverage Segment")
            lines.append(f"- Segment: {seg.get('segment_id')}")
            lines.append(f"- A/E: {seg.get('ae_ratio')}")

        if context.spotlight_segment:
            seg = context.spotlight_segment
            lines.append(f"\n## Spotlight Segment (Exception)")
            lines.append(f"- Segment: {seg.get('segment_id')}")
            lines.append(f"- Risk Level: {seg.get('risk_level')}")
            lines.append(f"- A/E: {seg.get('ae_ratio')}")

        lines.append(f"\n## Model Calibration")
        lines.append(f"- Overall A/E: {context.overall_ae:.4f}")

        lines.append("\n---")
        lines.append("Generate structured actuarial interpretation based on the evidence above.")
        lines.append("Output must be valid JSON matching the ActuarialInsights schema.")
        lines.append(
            "\nREMINDER — HARD CONSTRAINTS (violation will be flagged by automated checker):\n"
            "- Data period is 2012-2019 ONLY. Do NOT reference COVID-19, pandemic, or any year >= 2020.\n"
            "- Use ONLY the mortality rate and A/E values shown above. Do not introduce other numeric estimates.\n"
            "- All required disclaimers (data limitation, ASOP, reference-only) MUST appear in output."
        )

        return "\n".join(lines)

    def _build_validation_context_prompt(self, overall_ae: dict, yearly_ae: dict) -> str:
        """Build user prompt for validation diagnostic LLM call."""
        lines = []
        lines.append("## Calibration Summary")
        ae_val = overall_ae.get("overall_ae", "N/A")
        lines.append(f"- Overall A/E: {ae_val}")
        lines.append(f"- Total Actual Deaths: {overall_ae.get('total_actual_deaths', 'N/A'):,}" if overall_ae.get('total_actual_deaths') else "- Total Actual Deaths: N/A")
        lines.append(f"- Data Period: {overall_ae.get('data_period', 'N/A')}")

        lines.append("\n## Yearly A/E Trend")
        for yr, ae in sorted(yearly_ae.items()):
            flag = ""
            if isinstance(ae, (int, float)):
                if abs(ae - 1.0) > 0.01:
                    flag = " &lt;--deviation > +/-1%"
            lines.append(f"- {yr}: A/E = {ae}{flag}")

        lines.append("\n---")
        lines.append(
            "Interpret A/E stability. Flag years deviating >+/-1% from 1.0. "
            "Suggest hypothesis directions using conditional language (may, might, could). "
            "Use ASOP 23/25/41/56 framing. Output valid JSON matching ExperienceTrendPackInsights schema."
        )
        return "\n".join(lines)

    def _self_review(
        self,
        system_prompt: str,
        user_prompt: str,
        draft,
        output_schema
    ):
        """
        Second-pass review: verify constraints and return corrected JSON only.
        """
        if not self.is_available():
            return draft

        try:
            if hasattr(draft, "model_dump"):
                draft_json = json.dumps(draft.model_dump(), ensure_ascii=True)
            else:
                draft_json = json.dumps(draft, ensure_ascii=True)
        except Exception:
            draft_json = str(draft)

        review_instructions = (
            "You are reviewing a DRAFT actuarial JSON response. "
            "Check each item in the list below and fix any violations. "
            "Output ONLY valid JSON matching the required schema --no commentary.\n\n"
            "CHECK LIST (verify each item, fix if violated):\n"
            "1. All numeric values (A/E ratios, mortality rates) must EXACTLY match the context --do not invent numbers.\n"
            "2. No causal language: replace 'causes', 'proves', 'results in' with 'is associated with', 'coincides with', 'may reflect'.\n"
            "3. summary must reference the exact predicted mortality rate and overall A/E from context.\n"
            "4. recommendations: maximum 3 items, each actionable and evidence-linked.\n"
            "5. For SpotlightInsights: each segment_id must exactly match one of the input anomaly IDs.\n"
            "6. For TrendPackInsights: hypothesis_directions must use conditional language (may, might, could).\n"
            "7. All ASOP references must use correct numbers: 23 (data quality), 25 (credibility), "
            "41 (actuarial communication), 56 (modeling).\n"
            "8. No markdown, no code fences --output raw JSON only."
        )

        review_prompt = (
            f"{system_prompt}\n\n---\n\n{user_prompt}\n\n---\n\n"
            f"{review_instructions}\n\nDRAFT_JSON:\n{draft_json}"
        )

        try:
            # Use zero-temperature review client for deterministic constraint checking
            review_client = self._review_client if self._review_client is not None else None
            return self._invoke_with_timeout(review_prompt, output_schema, client=review_client)
        except Exception as e:
            print(f"[WARN] Self-review failed: {e}")
            return draft

    def _fallback_generate(self, prompt: str):
        """Fallback: regular generation with JSON parsing."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self.client.invoke, prompt)
                response = future.result(timeout=LLM_TIMEOUT_SECONDS)
            content = response.content

            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
            elif "{" in content:
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
            else:
                raise ValueError("No JSON found in response")

            data = json.loads(json_str)
            return ActuarialInsights(**data)
        except Exception as e:
            print(f"[WARN] Fallback parsing failed: {e}")
            return self._mock_insights(None)

    def _mock_insights(self, context) -> ActuarialInsights:
        """
        Placeholder insights skeleton when LLM is unavailable.
        is_placeholder=True signals downstream that this is not real LLM output.
        """
        from .schemas import SHAPExplanation

        return ActuarialInsights(
            summary="[PLACEHOLDER --LLM not available. Run with valid API key for real insights.]",
            risk_interpretation="[PLACEHOLDER --mortality rate and risk interpretation require LLM.]",
            shap_explanations=[
                SHAPExplanation(
                    feature="Attained_Age",
                    value="N/A",
                    direction="Increases",
                    explanation="[PLACEHOLDER --SHAP explanation requires LLM.]"
                )
            ],
            calibration_note="[PLACEHOLDER --calibration note requires LLM.]",
            segment_insight=None,
            recommendations=[
                "[PLACEHOLDER --recommendations require LLM. Check API key configuration.]"
            ]
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = LLM_MAX_RETRIES
    ) -> str:
        """Legacy method: Generate unstructured text response."""
        if not self.is_available():
            return "[LLM not available]"

        combined_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"

        for attempt in range(max_retries):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(self.client.invoke, combined_prompt)
                    response = future.result(timeout=LLM_TIMEOUT_SECONDS)
                return response.content
            except Exception as e:
                print(f"[WARN] LLM call failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return f"[LLM Error: {e}]"

        return "[LLM call failed]"


def create_llm_client(
    model_name: str = "gemini-2.5-flash-lite",
    temperature: float = 0.3,
    self_review: bool = True
) -> LLMClient:
    """Factory function to create and initialize LLM client."""
    return LLMClient(model_name, temperature, self_review=self_review).initialize()
