"""Pydantic v2 schema for Singapore insurance policy extraction.

This module is the single source of truth for the extraction schema. Both the
Gemini parser (as ``response_schema``) and the evaluation harness import this
model — neither duplicates the field definitions.

The verbose ``Field`` descriptions are intentional: Gemini reads them through
the structured-output ``response_schema``, so they double as per-field
extraction instructions.
"""

from typing import Literal

from pydantic import BaseModel, Field


class InsurancePolicyExtraction(BaseModel):

    policy_name: str = Field(
        description="""The official product name of the insurance policy as printed
        on the document. This is the plan name, not the insurer company name and
        not the policy number. Examples: 'PruActive Term', 'AIA Protect Plus',
        'Income TermLife Saver'. Do not include reference codes or policy numbers."""
    )

    insurer: str = Field(
        description="""The full legal name of the insurance company issuing this
        policy. Found in document header, registered details, or footer. Return
        company name only, not product name. Examples: 'Prudential Assurance Co.
        Singapore (Pte) Ltd.', 'AIA Singapore Private Limited', 'NTUC Income
        Insurance Co-operative Limited', 'Great Eastern Life Assurance Co. Ltd.',
        'Manulife (Singapore) Pte. Ltd.'"""
    )

    sum_assured_sgd: float | None = Field(
        description="""The primary life (death) coverage amount in Singapore dollars
        — the lump sum payable to beneficiaries upon the death of the life assured.

        LIFE COVERAGE ONLY. Do not return any of the following even if they appear
        as dollar amounts: critical illness benefit amounts, disability benefit
        amounts, premium waiver benefit amounts (these are not payouts), rider or
        supplementary benefit amounts, or sub-limits.

        May be labelled as: 'Death Benefit', 'Death Benefit Sum Assured', 'Life
        Coverage Amount', 'Basic Sum Insured', 'Sum Assured for Death', 'Face
        Amount', or 'Insured Amount'.

        Return as a numeric float only — no currency symbols, commas, or text.
        Example: 100000.00

        Return null if this document has no life or death coverage (e.g., a
        standalone hospitalisation or personal accident policy).""",
        default=None,
    )

    annual_premium_sgd: float | None = Field(
        description="""The total annual premium payable for this policy in
        Singapore dollars.

        Frequency conversion — apply before returning:
        - Monthly stated: multiply by 12
        - Quarterly stated: multiply by 4
        - Half-yearly stated: multiply by 2
        - Annual stated: return as-is

        Use the CURRENT premium. If the document states a premium that reduces or
        changes at a future date, extract the current premium only.

        Return gross premium before any CPF or SRS deductions. Return as a numeric
        annual float — no symbols or commas. Example: 676.44

        Return null only if no premium information is stated anywhere.""",
        default=None,
    )

    renewal_date: str | None = Field(
        description="""The date on which the policy expires or must be renewed,
        in ISO 8601 format (YYYY-MM-DD).

        Interpret by policy type:
        - Term life: return the cover end date (when the policy terminates)
        - Integrated Shield Plan / annual hospitalisation: return next renewal date
        - Whole life: return null — no fixed end date
        - Endowment: return the maturity date

        May be labelled as: 'Cover End Date', 'Policy Expiry Date', 'Renewal
        Date', 'Maturity Date', or 'Expiry Date'.

        Do NOT return the next premium due date — that is a payment schedule, not
        a renewal date.

        Return null if the policy has no fixed end or renewal date.""",
        default=None,
    )

    exclusions: list[str] = Field(
        description="""A list of conditions, events, or circumstances explicitly
        excluded from coverage. Each array item is one distinct exclusion as a
        concise complete sentence.

        SEARCH THE ENTIRE DOCUMENT. Singapore insurers structure exclusions
        differently:
        - Some have a dedicated 'General Exclusions' or 'Standard Exclusions'
          section
        - Others embed exclusions within each benefit section under headings like
          'What is not covered', 'Exclusions', or 'We do not pay if'
        - A document may have both general AND benefit-specific exclusions —
          capture all

        If the same exclusion appears across multiple benefit sections, include it
        once in deduplicated form.

        Do NOT include: waiting periods, claim submission procedures, documentary
        requirements, or payment conditions.

        If a Policy Schedule or Certificate contains a 'Special Exclusion' field
        with a stated value, include it. If it states 'Not Applicable', no
        policyholder-specific exclusions were added at underwriting — standard
        product exclusions from the T&C document still apply.

        Return empty array [] if no exclusions found. Never return null.""",
        default=[],
    )

    coverage_gaps: list[str] = Field(
        description="""A list of protection areas where this policy does NOT
        provide coverage, inferred against Singapore MAS and LIA benchmarks.

        Singapore benchmarks to use as reference:
        1. Life (death) coverage: LIA recommends 9-10x annual income. Note if
           life coverage is absent.
        2. Critical illness lump-sum payout: LIA recommends ~3.9x annual income.
           CRITICAL DISTINCTION: a CI Waiver benefit only waives future premiums
           — it does NOT pay a lump sum on CI diagnosis. A policy with CI Waiver
           but no CI Cover has a critical illness payout gap.
        3. Hospitalisation: MediShield Life is a baseline only. An Integrated
           Shield Plan (IP) is recommended. Note if hospitalisation coverage is
           absent.
        4. Total and permanent disability: Note if TPD coverage is absent.
        5. Income protection: Note if no disability income or income replacement
           benefit is provided.

        State gaps in terms of coverage TYPE — do not calculate income multiples
        as income is not in the document.

        Valid examples:
        - 'No critical illness lump-sum payout — policy includes CI premium
          waiver only, which waives premiums but does not pay a lump sum on
          CI diagnosis'
        - 'No hospitalisation or medical expense coverage'
        - 'No income protection or disability income benefit'

        Return empty array [] if policy covers all major protection areas.
        Do not fabricate gaps not clearly inferrable from the document.""",
        default=[],
    )

    policy_type: Literal[
        "life",
        "critical_illness",
        "hospitalisation",
        "income_protection",
        "personal_accident",
        "other",
    ] = Field(
        description="""The policy's PRIMARY coverage category, classified into
        exactly one of the six allowed values. Classify from policy_type_raw and
        the document's benefit structure:

        - 'life': primary benefit is a death benefit / life sum assured — term
          life, whole life, endowment, and investment-linked plans with a death
          benefit all belong here
        - 'critical_illness': primary benefit is a lump-sum payout on critical
          illness diagnosis (a CI premium waiver alone does NOT qualify)
        - 'hospitalisation': hospital/medical expense coverage — Integrated
          Shield Plans and their riders/co-pay supplements belong here
        - 'income_protection': disability income or income replacement benefit
          is the primary coverage
        - 'personal_accident': coverage is triggered by ACCIDENTS only —
          accidental death, accidental disability, accident medical
          reimbursement. A policy paying an 'accidental death benefit' is
          personal_accident, NOT life: it does not cover natural death.
        - 'other': the policy does not cleanly fit any category above

        Classify by the PRIMARY / base coverage, not by riders attached to it.""",
    )

    policy_type_raw: str | None = Field(
        description="""The policy's own stated name/type as the document
        describes itself, verbatim or close to it. Examples: 'Whole Life
        Participating Plan with CI Rider', 'Integrated Shield Plan — Premier',
        'PruPersonal Accident'. Extract this BEFORE classifying policy_type.

        Always populate this for every policy — especially when policy_type is
        'other', this field must still carry the document's own description.
        Return null only if the document truly states no plan name or type
        anywhere.""",
        default=None,
    )
