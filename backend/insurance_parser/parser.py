"""Gemini-powered extraction for Singapore insurance policy PDFs.

Uses the ``google-genai`` SDK (``from google import genai``). The parser sends the
raw PDF bytes to Gemini (no manual text extraction) with structured-output JSON
mode driven by the Pydantic schema, then validates the response. Recoverable
failures route through a reflection retry that feeds the error back to the model
— up to two retries.
"""

import json
import logging
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import ValidationError

from insurance_parser.prompts import INSURANCE_SYSTEM_PROMPT
from insurance_parser.schema import InsurancePolicyExtraction

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
_PDF_MIME_TYPE = "application/pdf"

# Our 9-field extraction is small (~2-3 KB of JSON), so 4096 tokens is generous
# headroom. It also caps the occasional repetition loop where the model
# over-generates until the model's much larger default ceiling.
MAX_OUTPUT_TOKENS = 4096


def _hit_max_tokens(response) -> bool:
    """True if the response was truncated because it hit ``MAX_TOKENS``.

    Tolerant of empty candidates and of the finish_reason being either a
    ``types.FinishReason`` enum (with ``.name``) or its raw integer value
    (2 == MAX_TOKENS).
    """
    try:
        candidate = response.candidates[0]
    except (AttributeError, IndexError, TypeError):
        return False
    finish_reason = getattr(candidate, "finish_reason", None)
    name = getattr(finish_reason, "name", None)
    if name is not None:
        return name == "MAX_TOKENS"
    return finish_reason == 2


class GeminiInsuranceParser:
    """Extracts :class:`InsurancePolicyExtraction` from a policy PDF via Gemini."""

    def __init__(self, api_key: str, model: str = "gemini-3.5-flash"):
        if not api_key or api_key == "your_key_here":
            raise ValueError(
                "A valid GEMINI_API_KEY is required. Set it in backend/.env "
                "(it is currently empty or the placeholder 'your_key_here')."
            )
        self._client = genai.Client(api_key=api_key)
        self.model_name = model
        # The new SDK builds the response schema directly from the Pydantic model
        # (its types.Schema supports field defaults), so no manual schema
        # post-processing is needed.
        self._config = types.GenerateContentConfig(
            system_instruction=INSURANCE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=InsurancePolicyExtraction,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        # Number of reflection retries used by the most recent extract() call
        # (0 = succeeded on the first attempt). Read by the evaluator.
        self.last_retries = 0

    def extract(self, pdf_path: str) -> InsurancePolicyExtraction | None:
        """Extract structured policy data from a PDF file on disk.

        Thin wrapper over :meth:`extract_bytes` for callers that have a file
        path (e.g. the evaluation harness).
        """
        path = Path(pdf_path)
        return self.extract_bytes(path.read_bytes(), filename=path.name)

    def extract_bytes(
        self, pdf_bytes: bytes, filename: str = "<in-memory PDF>"
    ) -> InsurancePolicyExtraction | None:
        """Extract structured policy data from in-memory PDF bytes.

        ``filename`` is used only for log messages. Returns a validated
        :class:`InsurancePolicyExtraction`, or ``None`` if the model keeps
        producing unusable output after the maximum number of retries.

        Recoverable failures — Pydantic ``ValidationError``, malformed/truncated
        JSON (``JSONDecodeError``), and blocked or empty responses (``ValueError``
        when the response has no text) — all route through the reflection retry.
        Only the retry budget is spent on them; unexpected errors propagate.
        """
        error_message = ""
        for attempt in range(MAX_RETRIES + 1):
            logger.info("Extraction attempt %d for %s", attempt, filename)
            try:
                if attempt == 0:
                    raw = self._call_gemini(pdf_bytes)
                else:
                    raw = self._validate_and_retry(pdf_bytes, error_message, attempt)
                result = InsurancePolicyExtraction.model_validate(raw)
                self.last_retries = attempt
                logger.info(
                    "Extraction succeeded for %s on attempt %d", filename, attempt
                )
                return result
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                error_message = self._format_error(exc)
                logger.warning(
                    "Attempt %d for %s failed: %s", attempt, filename, error_message
                )

        self.last_retries = MAX_RETRIES
        logger.error(
            "Extraction failed for %s after %d retries — giving up",
            filename,
            MAX_RETRIES,
        )
        return None

    def _call_gemini(self, pdf_bytes: bytes, extra_instruction: str = "") -> dict:
        """Call Gemini with the PDF and return the parsed JSON object as a dict.

        Raises ``ValueError`` if the response was blocked/empty (no text) and
        ``json.JSONDecodeError`` if the text is not valid JSON — both handled by
        :meth:`extract` as recoverable, retryable failures.
        """
        user_text = (
            "Extract the structured policy information from the attached PDF and "
            "return it as JSON matching the schema."
        )
        if extra_instruction:
            user_text = f"{user_text}\n\n{extra_instruction}"

        pdf_part = types.Part.from_bytes(data=pdf_bytes, mime_type=_PDF_MIME_TYPE)
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=[pdf_part, user_text],
            config=self._config,
        )
        if _hit_max_tokens(response):
            logger.warning(
                "Gemini response hit MAX_TOKENS (truncated at %d output tokens) - "
                "output is incomplete and will trigger a retry. Frequent "
                "occurrences across the gold set indicate the model is "
                "over-generating for these documents.",
                MAX_OUTPUT_TOKENS,
            )
        text = response.text
        if not text:
            raise ValueError("response had no text content (blocked or empty)")
        return json.loads(text)

    def _validate_and_retry(
        self, pdf_bytes: bytes, error: str, attempt: int
    ) -> dict:
        """Re-call Gemini, feeding the previous error back to the model."""
        logger.info("Reflection retry %d after error: %s", attempt, error)
        correction = (
            f"Your previous response failed with this error: {error}. Please "
            "correct it and return a single valid JSON object matching the schema."
        )
        return self._call_gemini(pdf_bytes, extra_instruction=correction)

    @staticmethod
    def _format_error(exc: Exception) -> str:
        """Render a recoverable extraction error as a compact, single-line message."""
        if isinstance(exc, ValidationError):
            parts = []
            for err in exc.errors():
                location = ".".join(str(loc) for loc in err["loc"]) or "(root)"
                parts.append(f"{location}: {err['msg']}")
            return "; ".join(parts)
        if isinstance(exc, json.JSONDecodeError):
            return f"response was not valid JSON (possibly truncated): {exc}"
        return f"response had no usable content (blocked or empty): {exc}"
