import os
import pickle
import sys
import numpy as np
from src.Exception import CustomException
from sklearn.model_selection import GridSearchCV
#Evauluation Metrics
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.metrics import precision_recall_curve, auc,roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


from sklearn.preprocessing import LabelBinarizer

def evaluate_model(X_train, X_test, y_train, y_test, models: dict, params: dict):
    try:
        report = {}

        for model_name, model in models.items():
            print("="*60)
            print(f"Evaluating Model: {model_name}")

            param_grid = params[model_name]
            grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, error_score='raise')
            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            y_pred = best_model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            # ✅ ROC AUC for binary or multiclass
            if hasattr(best_model, "predict_proba"):
                y_proba = best_model.predict_proba(X_test)

                try:
                    if y_proba.shape[1] == 2:
                        # Binary classification
                        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        # Multiclass (just in case)
                        lb = LabelBinarizer()
                        y_test_binarized = lb.fit_transform(y_test)
                        roc_auc = roc_auc_score(y_test_binarized, y_proba, average="macro", multi_class="ovr")
                except Exception as e:
                    roc_auc = f"ROC AUC Error: {e}"
            else:
                roc_auc = "N/A"

            # BEFORE MODEL TRAINING
            print("Unique classes in y_train:", np.unique(y_train))
            print("Unique classes in y_test:", np.unique(y_test))


            print(f"Best CV Score     : {grid.best_score_:.4f}")
            print(f"Best Params       : {grid.best_params_}")
            print(f"Accuracy Score    : {acc:.4f}")
            print(f"F1 Score          : {f1:.4f}")
            print(f"ROC AUC Score     : {roc_auc}")
            print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
            print("Classification Report:\n", classification_report(y_test, y_pred))

            report[model_name] = {
                "best_params": grid.best_params_,
                "accuracy_score": acc,
                "f1_score": f1,
                "roc_auc_score": roc_auc,
                "model": best_model
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
        try:
          with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
        except Exception as e:
            raise CustomException(e,sys)
