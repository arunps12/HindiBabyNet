"""Fixed column names of the Nettskjema exports and the names of the two CDI forms."""

from __future__ import annotations

SUBMISSION_ID = "$submission_id"
CREATED_AT = "$created"
ANSWER_TIME_MS = "$answer_time_ms"
FORWARDED_FORM_ID = "$forwarded_to_form"
SUBMISSION_REFERENCE = "SUBMISSION_REFERENCE"
REFERENCE_ID = "Reference ID"

CHILD_SEX_COLUMN = "बच्चे का लिंग"

# Columns of an export that are not questionnaire items.
FORM_METADATA_COLUMNS = {
    SUBMISSION_ID,
    CREATED_AT,
    ANSWER_TIME_MS,
    FORWARDED_FORM_ID,
    SUBMISSION_REFERENCE,
    REFERENCE_ID,
}

# Names of the two CDI forms: CDI-1 (younger children) and CDI-2 (older children).
CDI1_FORM = "CDI-I"
CDI2_FORM = "CDI-II"