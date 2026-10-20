"""Insurance PDF parser evaluation pipeline for the Unify project (CCDS25-1170).

Extracts structured data from Singapore insurance policy PDFs using Gemini and
evaluates extraction quality with deterministic scoring (Tier 1) and a
G-Eval-style LLM-as-judge harness (Tier 2) — the LLM-as-judge idea from G-Eval,
but with direct 0 / 0.5 / 1 scores, not G-Eval's probability-weighted token scoring.
"""

from insurance_parser.schema import InsurancePolicyExtraction
from insurance_parser.prompts import INSURANCE_SYSTEM_PROMPT
from insurance_parser.parser import GeminiInsuranceParser

__all__ = [
    "InsurancePolicyExtraction",
    "INSURANCE_SYSTEM_PROMPT",
    "GeminiInsuranceParser",
]
