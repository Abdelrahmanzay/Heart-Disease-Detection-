import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def evaluate_model(
    name: str,
    model,
    X_test,
    y_test,
    results: list,
    results_dir: str = "results"
):
    """
    Evaluate a trained model and:
    - compute metrics
    - save confusion matrix plot
    - save ROC curve plot (if possible)
    - append metrics to results list
    """

    y_pred = model.predict(X_test)

    # --- Basic metrics ---
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)

    # --- Confusion Matrix Plot ---
    cm_dir = os.path.join(results_dir, "confusion_matrices")
    _ensure_dir(cm_dir)

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No Disease", "Disease"])
    plt.yticks(tick_marks, ["No Disease", "Disease"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # annotate counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    cm_path = os.path.join(cm_dir, f"{name}_cm.png")
    plt.savefig(cm_path)
    plt.close()

    # --- ROC & AUC ---
    roc_auc = None
    roc_path = None
    try:
        # Prefer predict_proba
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            y_score = None

        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = roc_auc_score(y_test, y_score)

            roc_dir = os.path.join(results_dir, "roc_curves")
            _ensure_dir(roc_dir)

            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {name}")
            plt.legend(loc="lower right")
            roc_path = os.path.join(roc_dir, f"{name}_roc.png")
            plt.tight_layout()
            plt.savefig(roc_path)
            plt.close()

    except Exception:
        # safe fallback if ROC fails
        roc_auc = None

    # Append summary
    results.append(
        {
            "model": name,
            "accuracy": acc,
            "roc_auc": roc_auc,
            "cm_path": cm_path,
            "roc_path": roc_path,
            "classification_report": report,
        }
    )

    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    if roc_auc is not None:
        print(f"ROC AUC : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(report)

    return results
