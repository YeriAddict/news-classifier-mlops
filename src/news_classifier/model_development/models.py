from abc import ABC, abstractmethod
import pickle
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from news_classifier.feature_engineering.featurizer import Featurizer
from news_classifier.utils.models import Claim
from news_classifier.utils.reader import load_claims_from_csv, load_labels_from_csv

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X_train: List[Claim], y_train: List[str]):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        pass

    @abstractmethod
    def save(self, path):
        pass

class RandomForestModel(BaseModel):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config

        self.featurizer = Featurizer()

        self.model = RandomForestClassifier(**self.config["params"])

    def __calculate_probabilities__(self, X: List[Claim]) -> np.array:
        features = self.featurizer.transform(X)
        return self.model.predict_proba(features)

    def fit(self, X_train: List[Claim], y_train: List[str]) -> None:
        with open("../../../cache/featurizer.pkl", "rb") as f:
            self.featurizer = pickle.load(f)
        features = self.featurizer.transform(X_train)
        self.model.fit(features, y_train)

    def evaluate(self, X_test: List[Claim], y_test: List[str]) -> Dict:
        probabilities = self.__calculate_probabilities__(X_test)
        y_pred = np.argmax(probabilities, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probabilities[:, 1])
        conf_mat = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = conf_mat.ravel()
        return {
            "F1": f1,
            "Accuracy": accuracy,
            "AUC": auc,
            "True Negative": tn,
            "False Negative": fn,
            "False Positive": fp,
            "True Positive": tp,
        }

    def predict(self, X):
        probabilities = self.__calculate_probabilities__(X)
        y_pred = np.argmax(probabilities, axis=1)
        label = y_pred[0]
        return label

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

if __name__ == "__main__":
    X_train = load_claims_from_csv("../../../datasets/clean/data/train_data.csv")
    X_val = load_claims_from_csv("../../../datasets/clean/data/val_data.csv")
    X_test = load_claims_from_csv("../../../datasets/clean/data/test_data.csv")
    y_train = load_labels_from_csv("../../../datasets/clean/labels/train_labels.csv")
    y_val = load_labels_from_csv("../../../datasets/clean/labels/val_labels.csv")
    y_test = load_labels_from_csv("../../../datasets/clean/labels/test_labels.csv")

    model = RandomForestModel()
    model.fit(X_train, y_train)
    test_results = model.evaluate(X_test, y_test)
    validation_results = model.evaluate(X_val, y_val)
    print(test_results)
    print(validation_results)

    model.save("../../../cache/random_forest_model.pkl")