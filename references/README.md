# References

This folder has the earlier scientific and technical documents of HindiBabyNet. They describe the methods and the decisions of the project. The scientific content is kept as it was written.

Some of these documents show commands and folder names of an earlier version of the code. For the present commands, use the main [README](../README.md) and [audio/README.md](../audio/README.md).

`xgb_backend.md` and `pipeline_specification.md` describe the XGBoost method with speaker diarization. The present method does not use diarization. It is described in section 5 of [audio/README.md](../audio/README.md).

| Document | Content |
| --- | --- |
| `audio_preparation.md` | how the recordings are prepared |
| `pipeline_specification.md` | specification of the audio processing steps |
| `vtc_backend.md` | speaker classification with VTC |
| `xgb_backend.md` | speaker classification with the XGBoost model |
| `textgrid_generation.md` | Praat TextGrid files from the classification results |
| `manual_annotation_protocol.md` | protocol for manual speaker labels |
| `model_evaluation_protocol.md` | protocol for the evaluation of the automatic classification |
| `HindiBabyNet_R_Analysis_Implementation_Spec.md` | specification of the statistical analysis of vocal input and child vocalizations |
| `Hindi_MCDI_Wordbank_Implementation_Spec.md` | specification of the Hindi CDI processing |

## Semitone reference of the pitch measures

The prosody workflow gives pitch in Hz and in semitones (ST). Semitones are calculated as

    ST = 12 * log2(F0 / reference)

The study uses the reference **100 Hz** (`pitch.semitone_reference_hz` in `prosody/params.yaml`). An earlier example configuration had the reference 1 Hz. The two references give different numbers for the mean and the median pitch in ST: the values with 1 Hz are higher by 12 * log2(100) = 79.73 ST. The pitch range in ST and all measures in Hz are the same with both references. Do not combine ST values that were calculated with different references.

## CDI completion rule

A CDI questionnaire is counted as completed when a response exists in the CDI-1 or CDI-2 export of Nettskjema and it is linked to the background form. Items without an answer are scored as 0. The Hindi CDI specification in this folder also describes an earlier check of unanswered items at the end of the form. This check is not used for inclusion.
