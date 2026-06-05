import pandas as pd

from hindibabynet_cdi.linking import build_participant_linkage, summarize_participant_linkage


def _forms(*, age_text: str = "12", age_group: str = "8-18 महीने", birthdate: str = "2024-09-01", forwarded_to_form: str = "539642") -> dict[str, pd.DataFrame]:
    consent = pd.DataFrame(
        [
            {
                "$submission_id": "consent-1",
                "$created": "2025-08-30T15:11:10+02:00",
                "मुझे ऊपर वर्णित परियोजना के संबंध में जानकारी प्राप्त हो गई है।": "हाँ",
                "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।": "हाँ",
                "fill_date": "2025-08-30",
                "सहमति देने के लिए धन्यवाद। क्या आपके बच्चे का जन्म भारत में हुआ है और क्या वह जन्म से अब तक भारत में ही रह रहा है?": "हाँ",
                "$answer_time_ms": "1000",
            }
        ]
    )
    eligibility = pd.DataFrame(
        [
            {
                "$submission_id": "eligibility-1",
                "$created": "2025-08-30T16:11:10+02:00",
                "क्या आपके बच्चे की मातृभाषा हिंदी है?": "हाँ",
                "क्या आपका बच्चा प्री-टर्म जन्मा है?": "नहीं",
                "क्या आपके बच्चे को बोलने, सुनने या देखने से संबंधित कोई समस्या है?": "नहीं",
                "क्या आपके बच्चे की आयु 8 से 36 महीने के बीच है?": "हाँ",
                "Reference ID": "consent-1",
                "$answer_time_ms": "1000",
            }
        ]
    )
    background = pd.DataFrame(
        [
            {
                "$submission_id": "background-1",
                "$created": "2025-09-01T15:06:43+02:00",
                "यदि लागू हो, तो आपके बच्चे की दूसरी भाषा क्या है?": "",
                "आपका बच्चा कितने प्रतिशत समय दूसरी भाषा सुनता है?": "",
                "यदि लागू हो, तो आपके बच्चे की तीसरी भाषा क्या है?": "",
                "आपका बच्चा कितने प्रतिशत समय तीसरी भाषा सुनता है?": "",
                "माता की वर्तमान शिक्षा स्तर:": "Bachelor",
                "पिता की वर्तमान शिक्षा स्तर:": "Bachelor",
                "other_education": "",
                "माँ कहाँ पली-बढ़ी हैं?": "भारत",
                "पिता कहाँ पले-बढ़े हैं?": "भारत",
                "यदि आपने &#34;कोई अन्य देश&#34; चुना है, तो कृपया वह(वे) देश बताएं।": "",
                "आप कहाँ रहते हैं?": "शहर",
                "माँ की मातृभाषा क्या है?": "हिंदी",
                "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया माता की अन्य भाषा बताएं।": "",
                "पिता की मातृभाषा क्या है?": "हिंदी",
                "यदि आपने &#39;अन्य&#39; या  हिंदी &#43; अन्य चुना है, तो कृपया पिता की अन्य भाषा बताएं।": "",
                "बच्चा माँ के संपर्क में कितने प्रतिशत समय रहता है?": "75%",
                "बच्चा पिता के संपर्क में कितने प्रतिशत समय रहता है?": "25%",
                "birthdate": birthdate,
                "बच्चे की उम्र कितने महीनों की है?": age_text,
                "बच्चे की आयु": age_group,
                "SUBMISSION_REFERENCE": "eligibility-1",
                "$answer_time_ms": "1000",
                "$forwarded_to_form": forwarded_to_form,
            }
        ]
    )
    cdi_8_18 = pd.DataFrame(
        [
            {
                "$submission_id": "cdi-younger-1",
                "$created": "2025-09-01T11:00:00+02:00",
                "SUBMISSION_REFERENCE": "background-1",
                "$answer_time_ms": "1000",
            }
        ]
    )
    cdi_19_36 = pd.DataFrame(columns=["$submission_id", "$created", "SUBMISSION_REFERENCE", "$answer_time_ms"])
    return {
        "consent": consent,
        "eligibility": eligibility,
        "background": background,
        "cdi_8_18": cdi_8_18,
        "cdi_19_36": cdi_19_36,
    }


def test_build_participant_linkage_links_full_chain() -> None:
    linkage = build_participant_linkage(_forms())
    row = linkage.iloc[0]

    assert row["participant_id"] == "HBN_eligibility-1"
    assert row["consent_submission_id"] == "consent-1"
    assert row["eligibility_submission_id"] == "eligibility-1"
    assert row["background_submission_id"] == "background-1"
    assert row["cdi_submission_id"] == "cdi-younger-1"
    assert row["cdi_form_id"] == "539642"
    assert row["questionnaire"] == "8_18"
    assert row["consent_status"] == "confirmed"
    assert row["eligibility_status"] == "eligible"
    assert bool(row["included_analysis"]) is True


def test_build_participant_linkage_keeps_missing_consent_link() -> None:
    forms = _forms()
    forms["consent"] = pd.DataFrame(columns=forms["consent"].columns)

    linkage = build_participant_linkage(forms)
    row = linkage.iloc[0]

    assert row["consent_status"] == "missing_link"
    assert bool(row["included_analysis"]) is True


def test_build_participant_linkage_marks_explicit_non_consent_excluded() -> None:
    forms = _forms()
    forms["consent"].loc[0, "मैं सहमति देता/देती हूँ कि मेरी गुमनाम जानकारी ऊपर वर्णित परियोजना में उपयोग की जाएगी।"] = "नहीं"

    linkage = build_participant_linkage(forms)
    row = linkage.iloc[0]

    assert row["consent_status"] == "not_given"
    assert bool(row["included_analysis"]) is False
    assert "consent_not_given" in row["exclusion_reason"]


def test_build_participant_linkage_uses_actual_cdi_form_over_forward_hint() -> None:
    forms = _forms(forwarded_to_form="539642", age_text="28", age_group="19-36 महीने", birthdate="2023-05-01")
    forms["cdi_8_18"] = pd.DataFrame(columns=["$submission_id", "$created", "SUBMISSION_REFERENCE", "$answer_time_ms"])
    forms["cdi_19_36"] = pd.DataFrame(
        [
            {
                "$submission_id": "cdi-older-1",
                "$created": "2025-09-02T11:00:00+02:00",
                "SUBMISSION_REFERENCE": "background-1",
                "$answer_time_ms": "1000",
            }
        ]
    )

    linkage = build_participant_linkage(forms)

    assert linkage.iloc[0]["questionnaire"] == "19_36"
    assert linkage.iloc[0]["cdi_submission_id"] == "cdi-older-1"
    assert linkage.iloc[0]["cdi_form_id"] == "539644"
    assert bool(linkage.iloc[0]["forwarded_form_mismatch"]) is True


def test_build_participant_linkage_flags_age_questionnaire_mismatch() -> None:
    linkage = build_participant_linkage(_forms(age_text="30", age_group="8-18 महीने", birthdate="2023-03-01"))
    row = linkage.iloc[0]

    assert bool(row["age_group_raw_vs_calculated_mismatch"]) is True
    assert bool(row["questionnaire_age_range_mismatch"]) is True
    assert "questionnaire_age_range_mismatch" in row["linkage_quality_flag"]


def test_build_participant_linkage_reports_orphan_eligibility_row() -> None:
    forms = _forms()
    forms["consent"] = pd.DataFrame(columns=forms["consent"].columns)
    forms["background"] = pd.DataFrame(columns=forms["background"].columns)
    forms["cdi_8_18"] = pd.DataFrame(columns=forms["cdi_8_18"].columns)

    linkage = build_participant_linkage(forms)
    row = linkage.iloc[0]
    summary = summarize_participant_linkage(linkage)

    assert row["participant_id"] == "HBN_consent-1"
    assert row["consent_status"] == "missing_link"
    assert bool(row["included_analysis"]) is False
    assert "missing_cdi" in row["exclusion_reason"]
    assert int(summary.loc[summary["metric"] == "participants_total", "value"].iloc[0]) == 1