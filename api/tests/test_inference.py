import pandas as pd

from api.services.inference import inference_service


def test_model_loaded():
    assert inference_service.model is not None
    assert inference_service.scaler is not None
    assert len(inference_service.feature_cols) > 0


def test_predict_single_returns_valid_probabilities():
    record = {col: 0 for col in inference_service.feature_cols}
    record["protocol_type"] = "tcp"
    record["service"] = "http"
    record["flag"] = "SF"

    result = inference_service.predict_single(record)
    probs = result["class_probabilities"]
    total = sum(probs.values())
    assert abs(total - 1.0) < 0.01
    assert result["predicted_category"] in probs


def test_predict_batch_matches_row_count():
    df = pd.DataFrame([
        {col: 0 for col in inference_service.feature_cols}
        for _ in range(5)
    ])
    for col in ("protocol_type", "service", "flag"):
        df[col] = "tcp" if col == "protocol_type" else ("http" if col == "service" else "SF")

    result_df = inference_service.predict_batch(df)
    assert len(result_df) == 5
    assert "predicted_category" in result_df.columns
    assert "severity" in result_df.columns


def test_unseen_categorical_value_does_not_crash():
    record = {col: 0 for col in inference_service.feature_cols}
    record["protocol_type"] = "totally_new_protocol"
    record["service"] = "totally_new_service"
    record["flag"] = "totally_new_flag"

    result = inference_service.predict_single(record)
    assert "predicted_category" in result
