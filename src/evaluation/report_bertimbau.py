import json
import os
import sys
from typing import Optional

LATEST_MODEL_FILE = "bertimbau_latest.txt"
DEFAULT_MODEL_DIR = "bertimbau_out"


def resolve_model_dir(cli_model_dir: Optional[str] = None) -> str:
    if cli_model_dir:
        return cli_model_dir

    base_path = os.path.dirname(os.path.abspath(__file__))
    latest_path = os.path.join(base_path, LATEST_MODEL_FILE)
    if os.path.exists(latest_path):
        with open(latest_path, "r", encoding="utf-8") as f:
            candidate = f.read().strip().lstrip("\ufeff")
        if candidate:
            if not os.path.isabs(candidate):
                candidate = os.path.join(base_path, candidate)
            if os.path.isdir(candidate):
                return candidate

    fallback = os.path.join(base_path, DEFAULT_MODEL_DIR)
    if os.path.isdir(fallback):
        return fallback

    raise SystemExit(
        "Erro: modelo não encontrado. Use model=CAMINHO ou execute treino para gerar "
        + LATEST_MODEL_FILE
        + "."
    )


def parse_kv_args(argv):
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def main():
    argv = sys.argv[1:]
    if len(argv) == 0:
        print(
            "Uso:\n"
            "  python report_bertimbau.py\n"
            "  python report_bertimbau.py model=bertimbau_out_20260415_150955"
        )

    kv = parse_kv_args(argv)
    model_dir = resolve_model_dir(kv.get("model"))

    metrics_path = os.path.join(model_dir, "metrics.json")
    report_path = os.path.join(model_dir, "classification_report.txt")
    labels_path = os.path.join(model_dir, "labels.txt")
    cm_path = os.path.join(model_dir, "confusion_matrix.png")

    print(f"Modelo: {model_dir}")

    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\nMetricas:")
        for k in [
            "eval_loss",
            "eval_accuracy",
            "eval_f1_macro",
            "eval_f1_weighted",
            "epoch",
        ]:
            if k in metrics:
                print(f"- {k}: {metrics[k]}")
    else:
        print("\nMetricas: metrics.json nao encontrado")

    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_count = sum(1 for line in f if line.strip())
        print(f"\nNumero de labels: {labels_count}")
    else:
        print("\nNumero de labels: labels.txt nao encontrado")

    print("\nArtefactos:")
    print(f"- metrics: {metrics_path} ({'ok' if os.path.exists(metrics_path) else 'missing'})")
    print(f"- report: {report_path} ({'ok' if os.path.exists(report_path) else 'missing'})")
    print(f"- confusion_matrix: {cm_path} ({'ok' if os.path.exists(cm_path) else 'missing'})")

    if os.path.exists(report_path):
        print("\nExcerto do classification_report:")
        with open(report_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 16:
                    break
                print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
