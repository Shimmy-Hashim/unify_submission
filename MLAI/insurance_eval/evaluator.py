"""Evaluation harness for the insurance PDF parser.

Two scoring tiers:

* **Tier 1 — deterministic** scoring of the six simple fields (precision /
  recall / F1, document-level accuracy, and fabrication rate on absent fields).
  ``policy_type`` is scored as a categorical field with a confusion breakdown;
  ``policy_type_raw`` is free text with no fixed ground truth and is excluded
  from F1 scoring (spot-checked qualitatively via the report columns instead).
* **Tier 2 — a G-Eval-style LLM judge** (``gemini-3.5-flash``) scoring the two
  list fields (``exclusions`` faithfulness, ``coverage_gaps`` accuracy) against
  the PDF text (extracted with ``pypdf`` for the judge only). This follows
  G-Eval's LLM-as-judge idea but NOT its full method: scores are direct 0 / 0.5 /
  1 judge outputs, with no probability-weighted token scoring and no
  auto-generated chain-of-thought evaluation steps.

Extraction and scoring are separable: extraction never needs gold annotations,
and ``score_only`` mode re-scores ``raw_extractions.json`` without re-running PDF
extraction (the Tier 2 judge still calls Gemini to score the two list fields).

CLI (run from anywhere; path defaults resolve relative to this file)::

    python MLAI/insurance_eval/evaluator.py --mode zero_shot
"""

import argparse
import json
import logging
import os
import re
import string
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

# This evaluation harness lives in MLAI/insurance_eval/; the production parser
# package it exercises lives in backend/. Make backend/ importable.
_HERE = Path(__file__).resolve().parent
_BACKEND_DIR = _HERE.parents[1] / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from insurance_parser.parser import GeminiInsuranceParser

logger = logging.getLogger(__name__)

# The six fields scored deterministically in Tier 1. policy_type_raw is
# deliberately absent: it is free text with no fixed ground truth, so it is
# spot-checked qualitatively, never F1-scored.
SIMPLE_FIELDS = [
    "policy_name",
    "insurer",
    "policy_type",
    "sum_assured_sgd",
    "annual_premium_sgd",
    "renewal_date",
]
TEXT_FIELDS = {"policy_name", "insurer"}
CATEGORICAL_FIELDS = {"policy_type"}  # strict enum — exact match after normalisation
NUMERIC_FIELDS = {"sum_assured_sgd", "annual_premium_sgd"}
DATE_FIELDS = {"renewal_date"}

# High-stakes fields tracked for fabrication (model invents a value that is not
# actually in the document).
FABRICATION_FIELDS = ["sum_assured_sgd", "renewal_date", "exclusions"]

NUMERIC_TOLERANCE = 0.01  # 1% relative tolerance
JUDGE_MODEL = "gemini-3.5-flash"
JUDGE_MAX_OUTPUT_TOKENS = 4096
TEMPLATE_FILENAME = "template.json"

SG_BENCHMARKS = (
    "Singapore LIA/MAS benchmarks: life (death) cover ~9-10x annual income; "
    "critical illness lump-sum payout ~3.9x annual income (a CI premium waiver "
    "is NOT a CI lump-sum payout); hospitalisation should be an Integrated "
    "Shield Plan beyond the MediShield Life baseline; total and permanent "
    "disability (TPD) cover; and income protection / disability income."
)


# --------------------------------------------------------------------------- #
# Value normalisation + matching helpers
# --------------------------------------------------------------------------- #

def _is_empty(value) -> bool:
    """True if a value counts as 'no value' (null / blank string / empty list)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return True
    return False


def _normalise_text(value: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace — for exact text match."""
    lowered = str(value).lower()
    stripped = lowered.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", stripped).strip()


def _normalise_date(value: str) -> str:
    """Normalise a date string to ISO ``YYYY-MM-DD`` where possible."""
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %B %Y", "%d %b %Y",
                "%B %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text  # fall back to the raw string


def _field_matches(field: str, predicted, gold) -> bool:
    """Apply the per-field matching rule (assumes both values are non-empty)."""
    if field in TEXT_FIELDS:
        return _normalise_text(predicted) == _normalise_text(gold)
    if field in CATEGORICAL_FIELDS:
        # Strict enum values — exact match, tolerant only of case/whitespace.
        return str(predicted).strip().lower() == str(gold).strip().lower()
    if field in NUMERIC_FIELDS:
        try:
            pred_f = float(predicted)
            gold_f = float(gold)
        except (TypeError, ValueError):
            return False
        if gold_f == 0:
            return pred_f == 0
        return abs(pred_f - gold_f) / abs(gold_f) < NUMERIC_TOLERANCE
    if field in DATE_FIELDS:
        return _normalise_date(predicted) == _normalise_date(gold)
    # Unreachable today: every SIMPLE_FIELDS entry belongs to one of the four sets above.
    # Kept as the safe default for a field added to SIMPLE_FIELDS without a matching type —
    # text comparison is the strictest of the four, so a new field fails loudly rather than
    # being scored leniently by accident.
    return _normalise_text(predicted) == _normalise_text(gold)


def _evaluate_field(field: str, predicted, gold) -> dict:
    """Classify one field as TP / FP / FN / TN and whether it is 'correct'.

    * gold present, predicted matches -> TP, correct
    * gold present, predicted empty   -> FN, incorrect
    * gold present, predicted wrong   -> FP, incorrect
    * gold empty,   predicted empty   -> TN, correct
    * gold empty,   predicted present -> FP, incorrect (fabrication)
    """
    gold_empty = _is_empty(gold)
    pred_empty = _is_empty(predicted)

    if gold_empty:
        if pred_empty:
            return {"outcome": "TN", "correct": True}
        return {"outcome": "FP", "correct": False}

    if pred_empty:
        return {"outcome": "FN", "correct": False}

    if _field_matches(field, predicted, gold):
        return {"outcome": "TP", "correct": True}
    return {"outcome": "FP", "correct": False}


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #

class InsurancePolicyEvaluator:
    """Loads gold annotations and scores model extractions across both tiers."""

    def __init__(
        self,
        gold_dir: Path,
        output_dir: Path,
        gemini_api_key: Optional[str] = None,
    ):
        self.gold_dir = Path(gold_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gemini_api_key = gemini_api_key
        self._judge = None  # lazily created google-genai client

    # ----- gold loading --------------------------------------------------- #

    def load_gold(self) -> dict:
        """Load every gold annotation file (keyed by filename), skipping template."""
        gold = {}
        if not self.gold_dir.exists():
            return gold
        for path in sorted(self.gold_dir.glob("*.json")):
            if path.name == TEMPLATE_FILENAME:
                continue
            try:
                # utf-8-sig tolerates a BOM from Windows editors; writes stay BOM-free.
                record = json.loads(path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed gold file %s: %s", path.name, exc)
                continue
            if not isinstance(record, dict):
                logger.warning(
                    "Skipping gold file %s: expected a JSON object, got %s",
                    path.name,
                    type(record).__name__,
                )
                continue
            filename = record.get("filename")
            if not filename:
                logger.warning("Gold file %s has no 'filename' — skipping", path.name)
                continue
            gold[filename] = record
        return gold

    # ----- Tier 2: G-Eval-style LLM judge (direct 0/0.5/1, not full G-Eval) --- #

    def _get_judge(self):
        if self._judge is None:
            self._judge = genai.Client(api_key=self.gemini_api_key)
        return self._judge

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        """Extract plain text from a PDF for the G-Eval judge (judge-only)."""
        from pypdf import PdfReader

        reader = PdfReader(str(pdf_path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)

    @staticmethod
    def _parse_judge_json(text: str) -> dict:
        """Pull a JSON object out of the judge's response (tolerant of fences)."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _judge_call(self, prompt: str) -> dict:
        response = self._get_judge().models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
                max_output_tokens=JUDGE_MAX_OUTPUT_TOKENS,
            ),
        )
        text = response.text
        if not text:
            raise ValueError("judge returned no text content (blocked or empty)")
        return self._parse_judge_json(text)

    def geval_exclusions(self, pdf_text: str, exclusions: list) -> Optional[dict]:
        """Faithfulness G-Eval for the exclusions list. Empty list -> None."""
        if _is_empty(exclusions):
            return None
        prompt = (
            "You are evaluating whether extracted insurance policy exclusions are "
            "faithfully grounded in the source document. For each exclusion in the "
            "list, score it: 1 = directly stated in document, 0.5 = related to "
            "document content but imprecise or overly generalised, 0 = not found in "
            "document or fabricated. Return a JSON object with keys: scores (list of "
            "floats), average (float), reasoning (list of strings explaining each "
            "score).\n\n"
            f"SOURCE DOCUMENT:\n{pdf_text}\n\n"
            f"EXTRACTED EXCLUSIONS:\n{json.dumps(exclusions, indent=2)}"
        )
        return self._judge_call(prompt)

    def geval_coverage_gaps(self, pdf_text: str, gaps: list) -> Optional[dict]:
        """Accuracy G-Eval for the coverage_gaps list. Empty list -> None."""
        if _is_empty(gaps):
            return None
        prompt = (
            "You are evaluating whether identified insurance coverage gaps are "
            "accurate given the policy document and Singapore LIA/MAS benchmarks. "
            "For each gap, score it: 1 = correctly identifies a missing coverage "
            "type based on document content and benchmarks, 0.5 = gap exists but "
            "imprecisely stated, 0 = coverage is actually present in the document "
            "or gap is fabricated. Return a JSON object with keys: scores (list of "
            "floats), average (float), reasoning (list of strings).\n\n"
            f"{SG_BENCHMARKS}\n\n"
            f"SOURCE DOCUMENT:\n{pdf_text}\n\n"
            f"IDENTIFIED COVERAGE GAPS:\n{json.dumps(gaps, indent=2)}"
        )
        return self._judge_call(prompt)

    # ----- scoring a single document ------------------------------------- #

    def score_document(
        self, gold: dict, extraction: Optional[dict], pdf_path: Optional[Path]
    ) -> dict:
        """Score one document against its gold annotation. Returns a result dict."""
        ground_truth = gold.get("ground_truth", {})
        absent_fields = gold.get("absent_fields", {})
        extraction = extraction or {}

        per_field = {}
        all_correct = True
        for field in SIMPLE_FIELDS:
            result = _evaluate_field(
                field, extraction.get(field), ground_truth.get(field)
            )
            per_field[field] = result
            if not result["correct"]:
                all_correct = False

        # Fabrication: only meaningful where gold marks the field genuinely absent.
        fabrication = {}
        for field in FABRICATION_FIELDS:
            if absent_fields.get(field) is True:
                fabrication[field] = not _is_empty(extraction.get(field))
            else:
                fabrication[field] = None  # not an absent-field case

        # Tier 2 G-Eval (only if a judge is configured and PDF text is available).
        exclusions_geval = None
        coverage_gaps_geval = None
        if self.gemini_api_key and pdf_path and pdf_path.exists():
            try:
                pdf_text = self._extract_pdf_text(pdf_path)
                exclusions_geval = self.geval_exclusions(
                    pdf_text, extraction.get("exclusions", [])
                )
                coverage_gaps_geval = self.geval_coverage_gaps(
                    pdf_text, extraction.get("coverage_gaps", [])
                )
            except Exception as exc:  # judge failures must not crash the run
                logger.warning(
                    "G-Eval failed for %s: %s", gold.get("filename"), exc
                )

        return {
            "per_field": per_field,
            "document_accuracy": all_correct,
            "fabrication": fabrication,
            "exclusions_geval": exclusions_geval,
            "coverage_gaps_geval": coverage_gaps_geval,
        }


# --------------------------------------------------------------------------- #
# Aggregation + report writing
# --------------------------------------------------------------------------- #

def _precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def _geval_average(geval: Optional[dict]):
    """Pull the numeric average out of a G-Eval result, or None."""
    if not geval:
        return None
    avg = geval.get("average")
    return avg if isinstance(avg, (int, float)) else None


def _write_reports(output_dir: Path, rows: list, summary: dict) -> None:
    report_path = output_dir / "evaluation_report.csv"
    pd.DataFrame(rows).to_csv(report_path, index=False)

    summary_path = output_dir / "field_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Wrote %s and %s", report_path.name, summary_path.name)


# --------------------------------------------------------------------------- #
# Extraction stage
# --------------------------------------------------------------------------- #

def _run_extraction(
    pdfs_dir: Path, output_dir: Path, gemini_api_key: str, model: str
) -> list:
    """Run Gemini extraction over every PDF and persist raw_extractions.json."""
    parser = GeminiInsuranceParser(api_key=gemini_api_key, model=model)
    extractions = []
    pdf_paths = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found in %s", pdfs_dir)
    for pdf_path in pdf_paths:
        logger.info("Extracting %s", pdf_path.name)
        # A hard failure on one PDF (rate limit, network, API error) must not
        # abort the whole batch or discard the extractions already completed.
        # Record it as a failure and carry on; raw_extractions.json is still
        # written below with every document processed so far.
        try:
            result = parser.extract(str(pdf_path))
        except Exception as exc:  # noqa: BLE001 — keep the batch resilient
            logger.error("Extraction errored for %s: %s", pdf_path.name, exc)
            result = None
        extractions.append(
            {
                "filename": pdf_path.name,
                "extraction": result.model_dump() if result else None,
                "retries": parser.last_retries,
                "failed": result is None,
            }
        )
    raw_path = output_dir / "raw_extractions.json"
    raw_path.write_text(json.dumps(extractions, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d documents)", raw_path.name, len(extractions))
    return extractions


def _load_raw_extractions(output_dir: Path) -> list:
    raw_path = output_dir / "raw_extractions.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run --mode zero_shot first to produce "
            "extractions before scoring with --mode score_only."
        )
    return json.loads(raw_path.read_text(encoding="utf-8-sig"))


# --------------------------------------------------------------------------- #
# Main entrypoint
# --------------------------------------------------------------------------- #

def run_evaluation(
    pdfs_dir: Path,
    gold_dir: Path,
    output_dir: Path,
    mode: str = "zero_shot",
    model: str = "gemini-3.5-flash",
) -> None:
    """Run the full pipeline: extract (optional) then score against gold."""
    pdfs_dir = Path(pdfs_dir)
    gold_dir = Path(gold_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(dotenv_path=_BACKEND_DIR / ".env")
    load_dotenv()  # also honour an .env in the current working directory
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    # The G-Eval judge reuses the Gemini key; treat the placeholder as unset.
    judge_api_key = (
        gemini_api_key
        if gemini_api_key and gemini_api_key != "your_key_here"
        else None
    )

    # --- extraction stage ------------------------------------------------- #
    if mode == "zero_shot":
        if not gemini_api_key or gemini_api_key == "your_key_here":
            raise SystemExit(
                "GEMINI_API_KEY is missing or still the placeholder. Set a real "
                "key in backend/.env to run zero_shot extraction."
            )
        extractions = _run_extraction(pdfs_dir, output_dir, gemini_api_key, model)
    elif mode == "score_only":
        extractions = _load_raw_extractions(output_dir)
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (expected zero_shot or score_only)")

    # --- gold loading ----------------------------------------------------- #
    evaluator = InsurancePolicyEvaluator(
        gold_dir=gold_dir,
        output_dir=output_dir,
        gemini_api_key=judge_api_key,
    )
    gold = evaluator.load_gold()
    if not gold:
        print(
            "No gold annotations found. Run extraction only or add gold "
            "annotations to gold_set/ to enable scoring."
        )
        return

    if not judge_api_key:
        logger.warning(
            "GEMINI_API_KEY not set — Tier 2 G-Eval will be skipped "
            "(exclusions/coverage_gaps scores will be null)."
        )

    # --- scoring stage ---------------------------------------------------- #
    field_counts = {f: {"tp": 0, "fp": 0, "fn": 0} for f in SIMPLE_FIELDS}
    fabrication_counts = {f: {"fab": 0, "total": 0} for f in FABRICATION_FIELDS}
    # gold policy_type -> {predicted policy_type -> count}; only off-diagonal
    # entries are misclassifications.
    policy_type_confusion = {}
    geval_totals = {
        "exclusions_faithfulness": [],
        "coverage_gaps_accuracy": [],
    }
    rows = []
    docs_all_correct = 0
    total_retries = 0
    extraction_failures = 0

    by_filename = {e["filename"]: e for e in extractions}

    for filename, gold_record in gold.items():
        entry = by_filename.get(filename)
        if entry is None:
            logger.warning(
                "Gold annotation for %s has no matching extraction — skipping",
                filename,
            )
            continue

        extraction = entry.get("extraction")
        # `or 0` because a failed extraction records retries as an explicit null, which
        # .get(..., 0) returns as None rather than substituting the default.
        retries = entry.get("retries", 0) or 0
        total_retries += retries
        if entry.get("failed") or extraction is None:
            extraction_failures += 1

        pdf_path = pdfs_dir / filename
        result = evaluator.score_document(gold_record, extraction, pdf_path)
        per_field = result["per_field"]

        # accumulate Tier 1 counts
        for field in SIMPLE_FIELDS:
            outcome = per_field[field]["outcome"]
            if outcome == "TP":
                field_counts[field]["tp"] += 1
            elif outcome == "FP":
                field_counts[field]["fp"] += 1
            elif outcome == "FN":
                field_counts[field]["fn"] += 1

        if result["document_accuracy"]:
            docs_all_correct += 1

        # accumulate policy_type confusion (gold -> predicted counts)
        predicted = extraction or {}
        gold_type = (gold_record.get("ground_truth", {}) or {}).get("policy_type")
        pred_type = predicted.get("policy_type") or "(missing)"
        if gold_type:
            confusion_row = policy_type_confusion.setdefault(gold_type, {})
            confusion_row[pred_type] = confusion_row.get(pred_type, 0) + 1

        # accumulate fabrication
        for field in FABRICATION_FIELDS:
            fab = result["fabrication"][field]
            if fab is not None:
                fabrication_counts[field]["total"] += 1
                if fab:
                    fabrication_counts[field]["fab"] += 1

        excl_avg = _geval_average(result["exclusions_geval"])
        gaps_avg = _geval_average(result["coverage_gaps_geval"])
        if excl_avg is not None:
            geval_totals["exclusions_faithfulness"].append(excl_avg)
        if gaps_avg is not None:
            geval_totals["coverage_gaps_accuracy"].append(gaps_avg)

        rows.append(
            {
                "policy_id": gold_record.get("policy_id", ""),
                "filename": filename,
                "policy_name_correct": per_field["policy_name"]["correct"],
                "insurer_correct": per_field["insurer"]["correct"],
                "policy_type_correct": per_field["policy_type"]["correct"],
                "policy_type_gold": gold_type,
                "policy_type_predicted": predicted.get("policy_type"),
                # Audit column for the qualitative spot-check — never F1-scored.
                "policy_type_raw_predicted": predicted.get("policy_type_raw"),
                "sum_assured_correct": per_field["sum_assured_sgd"]["correct"],
                "annual_premium_correct":
                    per_field["annual_premium_sgd"]["correct"],
                "renewal_date_correct": per_field["renewal_date"]["correct"],
                "document_accuracy": result["document_accuracy"],
                "exclusions_geval": excl_avg,
                "coverage_gaps_geval": gaps_avg,
                "sum_assured_fabricated": result["fabrication"]["sum_assured_sgd"],
                "renewal_date_fabricated": result["fabrication"]["renewal_date"],
                "exclusions_fabricated": result["fabrication"]["exclusions"],
                "retries_needed": retries,
            }
        )

    total_documents = len(rows)

    # --- aggregate summary ------------------------------------------------ #
    per_field_summary = {
        field: _precision_recall_f1(
            field_counts[field]["tp"],
            field_counts[field]["fp"],
            field_counts[field]["fn"],
        )
        for field in SIMPLE_FIELDS
    }

    fabrication_rates = {}
    for field in FABRICATION_FIELDS:
        total = fabrication_counts[field]["total"]
        fabrication_rates[field] = (
            fabrication_counts[field]["fab"] / total if total else None
        )

    def _mean(values):
        return sum(values) / len(values) if values else None

    summary = {
        "per_field": per_field_summary,
        "policy_type_confusion": policy_type_confusion,
        "document_accuracy": (
            docs_all_correct / total_documents if total_documents else 0.0
        ),
        "fabrication_rates": fabrication_rates,
        "geval_averages": {
            "exclusions_faithfulness": _mean(
                geval_totals["exclusions_faithfulness"]
            ),
            "coverage_gaps_accuracy": _mean(
                geval_totals["coverage_gaps_accuracy"]
            ),
        },
        "total_documents": total_documents,
        "total_retries": total_retries,
        "extraction_failures": extraction_failures,
    }

    _write_reports(output_dir, rows, summary)
    print(
        f"Scored {total_documents} document(s). "
        f"Document accuracy: {summary['document_accuracy']:.2%}. "
        f"Results written to {output_dir}/."
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    arg_parser = argparse.ArgumentParser(
        description="Evaluate the Gemini insurance PDF parser."
    )
    arg_parser.add_argument(
        "--pdfs",
        default=str(_HERE / "pdfs"),
        help="Directory of PDF files (defaults to pdfs/ next to this script).",
    )
    arg_parser.add_argument(
        "--gold",
        default=str(_HERE / "gold_set"),
        help="Directory of gold annotation JSON files "
             "(defaults to gold_set/ next to this script).",
    )
    arg_parser.add_argument(
        "--output",
        default=str(_HERE / "results"),
        help="Output directory for reports (defaults to results/ next to this script).",
    )
    arg_parser.add_argument(
        "--mode",
        default="zero_shot",
        choices=["zero_shot", "score_only"],
        help="zero_shot: extract with Gemini then score. "
        "score_only: re-score existing raw_extractions.json.",
    )
    arg_parser.add_argument(
        "--model",
        default="gemini-3.5-flash",
        help="Gemini model id used for extraction (zero_shot mode).",
    )
    args = arg_parser.parse_args()

    run_evaluation(
        pdfs_dir=Path(args.pdfs),
        gold_dir=Path(args.gold),
        output_dir=Path(args.output),
        mode=args.mode,
        model=args.model,
    )


if __name__ == "__main__":
    main()
