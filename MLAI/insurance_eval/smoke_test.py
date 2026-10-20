"""End-to-end smoke test for the insurance parser pipeline.

Generates a minimal one-page PDF as raw bytes (no third-party PDF library) with
fictional policy text, then runs a **live** Gemini extraction through
``GeminiInsuranceParser`` and asserts a valid ``InsurancePolicyExtraction`` is
returned.

Requires a real ``GEMINI_API_KEY`` in ``backend/.env``. The content below is
entirely fictional — it is NOT a real policy and contains no personal data.

Run from anywhere (paths resolve relative to this file)::

    python MLAI/insurance_eval/smoke_test.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# This smoke test lives in MLAI/insurance_eval/; the production parser package
# it exercises lives in backend/. Make backend/ importable.
_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(_BACKEND_DIR))

from insurance_parser.parser import GeminiInsuranceParser
from insurance_parser.schema import InsurancePolicyExtraction

# Fictional policy text — purely synthetic, for pipeline testing only.
DUMMY_POLICY_LINES = [
    "CERTIFICATE OF LIFE ASSURANCE",
    "Insurer: Specimen Assurance Singapore Pte. Ltd.",
    "Product Name: Specimen Term Protect",
    "Life Assured: A. Specimen",
    "Death Benefit Sum Assured: SGD 150,000",
    "Premium: SGD 40.00 per month, payable monthly",
    "Cover End Date: 31 December 2050",
    "General Exclusions:",
    "No benefit is payable for death by suicide within the first policy year.",
    "No benefit is payable for pre-existing conditions not declared at application.",
    "This plan provides death cover only. It does not include critical illness,",
    "hospitalisation, or disability income benefits.",
]


def _escape_pdf_text(text: str) -> str:
    """Escape the three characters that terminate a PDF string literal.

    Backslash MUST be replaced first — doing parens first would then double-escape the
    backslashes this function itself just introduced."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_minimal_pdf(lines: list[str]) -> bytes:
    """Assemble a valid single-page PDF with the given text lines."""
    # Helvetica 12pt, first baseline 50pt from the left and 770pt up an 842pt page, 16pt
    # line spacing; T* advances one line, so the lines simply stack downward.
    ops = ["BT", "/F1 12 Tf", "50 770 Td", "16 TL"]
    ops += [f"({_escape_pdf_text(line)}) Tj\nT*" for line in lines]
    ops.append("ET")
    content = "\n".join(ops).encode("latin-1")

    # ⚠️ Object numbers are positional: an object's number is its 1-based index in this
    # list, and the cross-references below are written out by hand to match. The page
    # object points at /Contents 4 0 R (the stream) and /F1 5 0 R (the font), so inserting
    # or reordering an entry here silently corrupts the PDF. Keep the list and the
    # references in step.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode("latin-1") + b" >>\nstream\n"
        + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{i} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"

    # The xref table always carries a leading free entry for object 0, so /Size counts one
    # more than the objects actually written.
    xref_offset = len(pdf)
    size = len(objects) + 1
    pdf += f"xref\n0 {size}\n".encode("latin-1")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets:
        pdf += f"{offset:010d} 00000 n \n".encode("latin-1")
    pdf += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(pdf)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load backend/.env — the parser package and its API key live in backend/.
    env_path = _BACKEND_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    load_dotenv()  # also honour an .env in the current working directory

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        print(
            "SMOKE TEST CANNOT RUN: GEMINI_API_KEY is missing or still the "
            "placeholder 'your_key_here'.\n"
            f"Set a real key in {env_path} and re-run:\n"
            "    python MLAI/insurance_eval/smoke_test.py"
        )
        return 1

    pdf_bytes = build_minimal_pdf(DUMMY_POLICY_LINES)
    print(f"Generated dummy PDF: {len(pdf_bytes)} bytes")

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "smoke_policy.pdf"
        pdf_path.write_bytes(pdf_bytes)

        parser = GeminiInsuranceParser(api_key=api_key)
        print("Calling Gemini for live extraction...")
        result = parser.extract(str(pdf_path))

    if result is None:
        print("SMOKE TEST FAILED: extraction returned None after retries.")
        return 1
    # Reported the same way as the None case rather than asserted: a bare assert would
    # crash without a message here, and disappears entirely under `python -O`.
    if not isinstance(result, InsurancePolicyExtraction):
        print(f"SMOKE TEST FAILED: extraction returned {type(result).__name__}, "
              f"not InsurancePolicyExtraction.")
        return 1

    print(f"SMOKE TEST PASSED (retries: {parser.last_retries}). Extraction:")
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
