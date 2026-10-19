from battery_soh_repro import data_pipeline as dp
from battery_soh_repro import feature_policy as fp


def test_frozen_strict_feature_counts():
    assert len(dp.FROZEN_STRICT_CONSTANT_FEATURES) == 335
    assert len(dp.STRICT_VARIABLE_FEATURES) == 314


def test_expected_feature_audit_counts():
    assert fp.EXPECTED_AUDIT_COUNTS["CONSTANT"] == {
        "STRICT_SAFE": 335,
        "PARTIAL_OBSERVATION": 105,
        "HISTORY_ONLY": 30,
        "INADMISSIBLE": 66,
        "REVIEW_REQUIRED": 85,
    }
    assert fp.EXPECTED_AUDIT_COUNTS["VARIABLE"] == {
        "STRICT_SAFE": 314,
        "PARTIAL_OBSERVATION": 91,
        "HISTORY_ONLY": 30,
        "INADMISSIBLE": 91,
        "REVIEW_REQUIRED": 46,
    }


def test_endpoint_capacity_features_are_not_strict():
    examples = [
        "LEGACY__Discharge_CapacityTo_3p2V_Ah",
        "INV__Discharge_EnergyTo_3p2V_Wh__pct",
        "INV__Discharge_TimeTo_3p2V_h__pct_ewm5",
    ]
    for feature in examples:
        classification = fp.classify_feature(feature, "VARIABLE")["Classification"]
        assert classification != fp.STRICT_SAFE
