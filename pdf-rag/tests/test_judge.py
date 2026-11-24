from pdfrag.judge import ClaimSupportJudge, ModelJudge, calibrate, judge_answer, load_calibration
from pdfrag.models import ScriptedModel
from pdfrag.pipeline import Pipeline


def test_claim_support_judge_agrees_with_hand_labels():
    labelled = load_calibration()
    result = calibrate(ClaimSupportJudge(), labelled)
    assert result["agreement"] >= 0.85
    assert result["false_fail"] == 0
    # its known blind spots: meaning, not words. These are why a judge is calibrated before it is trusted.
    misses = [i["claim"] for i in labelled if ClaimSupportJudge().supports(i["claim"], i["evidence"])[0] != i["supported"]]
    assert misses == ["The default polling interval is 5 seconds.", "Remote shell access is enabled by default."]


def test_judge_catches_an_invented_number(chunks):
    bad = "The maximum inlet pressure is 6 bar. [h200 p.1]"
    p = Pipeline(chunks, model=ScriptedModel([bad]))
    a = p.answer("What is the value for parameter Maximum inlet pressure on the H-200?")
    assert a.text == bad  # the citation is valid, so only the judge can catch it
    (verdict,) = judge_answer(a.text, a.retrieval.chunks, ClaimSupportJudge())
    assert not verdict.supported and "6" in verdict.reason


def test_model_judge_parses_strictly_and_calibrates():
    labelled = load_calibration()
    replies = ["SUPPORTED" if i["supported"] else "UNSUPPORTED." for i in labelled]
    assert calibrate(ModelJudge(ScriptedModel(replies)), labelled)["agreement"] == 1.0
    assert ModelJudge(ScriptedModel(["Probably supported"])).supports("x", "y")[0] is False
