import json
import os
import sys
from pathlib import Path
from typing import Optional

# Raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[2]

# Ficheiro que indica o último modelo treinado
LATEST_MODEL_FILE = ROOT_DIR / "results" / "bertimbau_latest.txt"

# Modelo utilizado por defeito caso não exista apontador
DEFAULT_MODEL_DIR = ROOT_DIR / "results" / "bertimbau_out"


# Determina qual o modelo a analisar
def resolve_model_dir(cli_model_dir: Optional[str] = None) -> str:

    # Caso seja fornecido explicitamente um modelo
    if cli_model_dir:
        return cli_model_dir

    # Tenta localizar automaticamente o último treino realizado
    if LATEST_MODEL_FILE.exists():

        with open(LATEST_MODEL_FILE, "r", encoding="utf-8") as f:
            candidate = f.read().strip().lstrip("\ufeff")

        if candidate:

            candidate_path = Path(candidate)

            if not candidate_path.is_absolute():
                candidate_path = ROOT_DIR / candidate

            # Verifica se a pasta do modelo existe
            if candidate_path.is_dir():
                return str(candidate_path)

    if DEFAULT_MODEL_DIR.is_dir():
        return str(DEFAULT_MODEL_DIR)

    raise SystemExit(
        "Erro: modelo não encontrado. "
        "Execute train_bertimbau.py para gerar um modelo."
    )

# Converte argumentos do formato:
# model=bertimbau_out_xxx
# para um dicionário Python
def parse_kv_args(argv):
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv


def main():
    argv = sys.argv[1:]
    # Mostra ajuda básica caso não sejam passados argumentos
    if len(argv) == 0:
        print(
            "Uso:\n"
            "  python report_bertimbau.py\n"
            "  python report_bertimbau.py model=bertimbau_out_20260415_150955"
        )

    kv = parse_kv_args(argv)
    model_dir = resolve_model_dir(kv.get("model"))

    # Principais artefactos produzidos pelo treino
    metrics_path = os.path.join(model_dir, "metrics.json")
    report_path = os.path.join(model_dir, "classification_report.txt")
    labels_path = os.path.join(model_dir, "labels.txt")
    cm_path = os.path.join(model_dir, "confusion_matrix.png")

    print(f"Modelo: {model_dir}")

    # Carrega métricas de avaliação do modelo
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        print("\nMetricas:")
        # Mostra apenas as métricas mais relevantes
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

    # Conta quantas categorias foram utilizadas no treino
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_count = sum(1 for line in f if line.strip())
        print(f"\nNumero de labels: {labels_count}")
    else:
        print("\nNumero de labels: labels.txt nao encontrado")

    # Verifica se todos os ficheiros esperados existem
    print("\nArtefactos:")
    print(f"- metrics: {metrics_path} ({'ok' if os.path.exists(metrics_path) else 'missing'})")
    print(f"- report: {report_path} ({'ok' if os.path.exists(report_path) else 'missing'})")
    print(f"- confusion_matrix: {cm_path} ({'ok' if os.path.exists(cm_path) else 'missing'})")

    # Apresenta um excerto rápido do relatório de classificação
    if os.path.exists(report_path):
        print("\nExcerto do classification_report:")
        with open(report_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                # Limita a saída para evitar imprimir o relatório completo
                if i >= 16:
                    break
                print(line.rstrip("\n"))


if __name__ == "__main__":
    main()
