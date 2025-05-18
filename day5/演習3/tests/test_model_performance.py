# day5/演習3/tests/test_model_performance.py
import sys, os
import json
import time
import pytest
from sklearn.metrics import accuracy_score

# テストから src を読めるようにパスを追加
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )
)

from model_utils import load_model, predict
from data_loader import load_test_data

# ベースライン指標を読み込むJSON
with open(os.path.join(os.path.dirname(__file__), "baseline_metrics.json")) as f:
    baseline = json.load(f)


@pytest.fixture(scope="module")
def model():
    # test ファイルから見たモデルファイルの絶対パスを計算
    model_fp = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),  # …/day5/演習3/tests/
            "..",                        # …/day5/演習3/
            "models",
            "current_model.pkl"
        )
    )
    return load_model(model_fp)


@pytest.fixture(scope="module")
def test_data():
    # tests ディレクトリから見たデータファイルの絶対パスを計算
    data_fp = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "Titanic.csv"
        )
    )
    return load_test_data(data_fp)


def test_accuracy_not_degraded(model, test_data):
    X_test, y_test = test_data
    preds = predict(model, X_test)
    acc = accuracy_score(y_test, preds)
    assert acc >= baseline["accuracy"], (
        f"Accuracy degraded: {acc:.3f} < baseline {baseline['accuracy']:.3f}"
    )


def test_inference_latency(model, test_data):
    X_test, _ = test_data
    _ = predict(model, X_test[:10])  # ウォームアップ

    start = time.perf_counter()
    _ = predict(model, X_test)
    elapsed = time.perf_counter() - start

    avg_latency = elapsed / len(X_test)
    assert avg_latency <= baseline["avg_latency"], (
        f"Latency increased: {avg_latency:.3f}s > baseline {baseline['avg_latency']:.3f}s"
    )
