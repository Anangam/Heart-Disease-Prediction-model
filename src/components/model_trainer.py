import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Logistic Regression": LogisticRegression(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC()
            }

            best_model_name = None
            best_score = 0
            results = {}

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = accuracy_score(y_test, y_pred)
                results[name] = round(score * 100, 2)
                logging.info(f"{name}: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = name

            logging.info(f"Best model: {best_model_name} with accuracy {best_score:.4f}")

            print("\n--- Model Results ---")
            for name, score in results.items():
                print(f"{name}: {score}%")

            return f"\nBest Model: {best_model_name} → Accuracy: {round(best_score * 100, 2)}%"

        except Exception as e:
            raise CustomException(e, sys)