from __future__ import annotations

SUBMISSION_ID = "$submission_id"
CREATED_AT = "$created"
ANSWER_TIME_MS = "$answer_time_ms"
FORWARDED_FORM_ID = "$forwarded_to_form"
SUBMISSION_REFERENCE = "SUBMISSION_REFERENCE"
REFERENCE_ID = "Reference ID"

CONSENT_INFORMATION_COLUMN = "मुझे ऊपर वर्णित परियोजना के संबंध में जानकारी प्राप्त हो गई है।"
CONSENT_RESPONSE_COLUMN = "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।"
INDIA_BIRTH_AND_RESIDENCE_COLUMN = (
    "सहमति देने के लिए धन्यवाद। क्या आपके बच्चे का जन्म भारत में हुआ है और क्या वह जन्म से अब तक भारत में ही रह रहा है?"
)

MOTHER_TONGUE_COLUMN = "क्या आपके बच्चे की मातृभाषा हिंदी है?"
PRETERM_COLUMN = "क्या आपका बच्चा प्री-टर्म जन्मा है?"
IMPAIRMENT_COLUMN = "क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?"
ELIGIBLE_AGE_RANGE_COLUMN = "क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?"

SECOND_LANGUAGE_COLUMN = "यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?"
SECOND_LANGUAGE_PERCENT_COLUMN = "आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?"
THIRD_LANGUAGE_COLUMN = "यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?"
THIRD_LANGUAGE_PERCENT_COLUMN = "आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?"
MOTHER_EDUCATION_COLUMN = "माता की वर्तमान शिक्षा स्तर:"
FATHER_EDUCATION_COLUMN = "पिता की वर्तमान शिक्षा स्तर:"
CHILD_SEX_COLUMN = "बच्चे का लिंग"
REPORTED_AGE_MONTHS_COLUMN = "बच्चे की उम्र कितने महीनों की है?"
BIRTHDATE_COLUMN = "birthdate"
CHILD_AGE_LABEL_COLUMN = "बच्चे की आयु"

CONTACT_MOBILE_COLUMN = (
    "<b>कृपया अपना व्हाट्सएप नंबर प्रदान करें, यदि आपके पास है, अन्यथा सामान्य फोन नंबर प्रदान करें।</b>"
)
CONTACT_EMAIL_COLUMN = "<b>Email address (ईमेल)</b>"

CDI1_COMPREHENSION_ONLY = "केवल समझता/समझती है"
CDI1_COMPREHENSION_AND_PRODUCTION = "समझता/समझती है और कहता/कहती है"
CDI2_PRODUCTION = "कहता/कहती है"

FORM_METADATA_COLUMNS = {
    SUBMISSION_ID,
    CREATED_AT,
    ANSWER_TIME_MS,
    FORWARDED_FORM_ID,
    SUBMISSION_REFERENCE,
    REFERENCE_ID,
}

PARTICIPANT_ID_COLUMNS = {
    "consent_submission_id",
    "eligibility_submission_id",
    "background_submission_id",
    "cdi_submission_id",
    "contact_submission_id",
    "eligibility_to_consent_id",
    "background_to_eligibility_id",
    "cdi_to_background_id",
    "contact_to_cdi_id",
}