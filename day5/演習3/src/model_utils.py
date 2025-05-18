# day5/演習3/src/model_utils.py
import pickle


def load_model(path="models/current_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(model, X):
    # X は pandas.DataFrame などを想定
    return model.predict(X)
