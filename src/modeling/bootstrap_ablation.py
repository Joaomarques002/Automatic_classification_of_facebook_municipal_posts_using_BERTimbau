import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Definição profissional de caminhos relativos ao projeto
MODELING_DIR = Path(__file__).resolve().parent
ROOT_DIR = MODELING_DIR.parent.parent
BASE_DIR = ROOT_DIR / "data" / "processed"
# Diretoria onde serão guardados os resultados estatísticos
RESULTS_DIR = ROOT_DIR / "results" / "bootstrap_results"

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
SEED = 42
TEXT_COLUMN = "post_text"

# Configuração dos dois estudos avaliados na dissertação
STUDIES = {
    # Dataset original utilizado nos experimentos iniciais
    "pilot": {
        "file": "dataset_labels_normalizado.csv",
        "label": "new_sector_norm",
        "frozen": "results/bertimbau_ablation_frozen", # Caminhos mapeados para a nova pasta results/
        "finetune": "results/bertimbau_ablation_finetune",
    },
    # Dataset expandido utilizado na fase final
    "expanded": {
        "file": "dataset_new_merged_train_ready.csv",
        "label": "CategoryName",
        "frozen": "results/bertimbau_ablation_newdata_frozen",
        "finetune": "results/bertimbau_ablation_newdata_finetune",
    },
}

# Permite escolher o estudo e o número de repetições Bootstrap
def parse_args():
    values = {"study": "pilot", "n": "1000"}  # Alinhado com o teu dataset padrão padrão
    for argument in sys.argv[1:]:
        if "=" in argument:
            key, value = argument.split("=", 1)
            values[key.strip().lower()] = value.strip()
    if values["study"] not in STUDIES:
        raise SystemExit("❌ Erro: o parâmetro 'study' deve ser 'pilot' ou 'expanded'")
    return values["study"], int(values["n"])

# Reconstrói o conjunto de teste utilizado na avaliação
def load_test_data(config):
    data_path = BASE_DIR / config["file"]
    if not data_path.exists():
        raise SystemExit(f"❌ Erro: O ficheiro de dados não foi encontrado em: {data_path}")
        
    df = pd.read_csv(data_path, encoding="utf-8")
    # Remove exemplos inválidos antes da avaliação
    df = df.dropna(subset=[TEXT_COLUMN, config["label"]]).copy()
    df = df[~df[TEXT_COLUMN].astype(str).str.startswith("[ERRO")]
    df = df.drop_duplicates(subset=[TEXT_COLUMN]).copy()
    if config["label"] == "CategoryName":
        df = df[df[config["label"]].astype(str) != "Geral"].copy()

    counts = df[config["label"]].astype(str).value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df[config["label"]].astype(str).isin(valid_classes)].copy()
    
    # Mantém a mesma divisão treino/teste utilizada no treino
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df[config["label"]].astype(str),
    )
    labels = sorted(df[config["label"]].astype(str).unique())
    # Converte labels textuais para IDs numéricos
    label_to_id = {label: index for index, label in enumerate(labels)}
    y_true = np.array([label_to_id[label] for label in test_df[config["label"]].astype(str)])
    return test_df[TEXT_COLUMN].astype(str).tolist(), y_true, labels

# Gera previsões utilizando um modelo previamente treinado
def predict(model_dir, texts):
    # Procura o modelo absoluto ou relativo à raiz do projeto
    full_model_path = ROOT_DIR / model_dir if not os.path.isabs(model_dir) else Path(model_dir)
    
    # Procura automaticamente o modelo mais recente
    # compatível com o prefixo indicado
    if not full_model_path.exists():
        parent_dir = full_model_path.parent
        matches = sorted([p for p in parent_dir.glob(f"{full_model_path.name}*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            full_model_path = matches[0]
        else:
            print(f"⚠️ Aviso: Caminho do modelo não encontrado: {full_model_path}. O script vai tentar usar pesos padrão.")
            full_model_path = MODEL_NAME

    print(f"🤖 A carregar pesos para inferência de: {full_model_path.name}")
    tokenizer = AutoTokenizer.from_pretrained(str(full_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(full_model_path))
    model.eval()
    
    # Utiliza GPU sempre que disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predictions = []
    batch_size = 64
    with torch.no_grad():
        # Executa inferência em lotes
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                texts[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
    return np.array(predictions)

# Métricas utilizadas
def metrics(y_true, predictions):
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
    }

# Calcula diferenças de desempenho através de Bootstrap
def bootstrap(y_true, frozen, finetune, repetitions):
    rng = np.random.default_rng(SEED)
    sample_count = len(y_true)
    values = np.empty((repetitions, 3), dtype=float)
    for index in range(repetitions):
        # Gera uma nova amostra Bootstrap com reposição
        sample = rng.integers(0, sample_count, size=sample_count)
        frozen_metrics = metrics(y_true[sample], frozen[sample])
        finetune_metrics = metrics(y_true[sample], finetune[sample])
        # Calcula o ganho do fine-tuning face ao encoder congelado
        values[index] = [
            finetune_metrics[metric] - frozen_metrics[metric]
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        ]
    # Estima diferença média e intervalo de confiança    
    return {
        metric: {
            "difference": float(values[:, column].mean()),
            "ci95_low": float(np.percentile(values[:, column], 2.5)),
            "ci95_high": float(np.percentile(values[:, column], 97.5)),
            "p_bootstrap_difference_le_zero": float(np.mean(values[:, column] <= 0)),
        }
        for column, metric in enumerate(("accuracy", "macro_f1", "weighted_f1"))
    }


def main():
    study, repetitions = parse_args()
    config = STUDIES[study]
    
    print(f"📊 A iniciar análise bootstrap para o estudo '{study}' ({repetitions} repetições)...")
    texts, y_true, labels = load_test_data(config)
    
    # Carrega os dois modelos que serão comparados
    frozen = predict(config["frozen"], texts)
    finetune = predict(config["finetune"], texts)

    # Calcula desempenho observado de cada configuração
    observed = {
        "frozen": metrics(y_true, frozen),
        "finetune": metrics(y_true, finetune),
    }
    result = {
        "study": study,
        "test_samples": len(y_true),
        "labels": labels,
        "seed": SEED,
        "bootstrap_repetitions": repetitions,
        "observed_metrics": observed,
        # Calcula robustez estatística das diferenças observadas
        "bootstrap_difference_finetune_minus_frozen": bootstrap(
            y_true, frozen, finetune, repetitions
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # Guarda relatório completo para futura análise
    (RESULTS_DIR / f"{study}_bootstrap.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Guarda previsões dos dois modelos para auditoria
    pd.DataFrame(
        {
            "y_true": y_true,
            "pred_frozen": frozen,
            "pred_finetune": finetune,
        }
    ).to_csv(RESULTS_DIR / f"{study}_predictions.csv", index=False)
    
    print("\n✅ Resumo do Bootstrap estatístico concluído:")
    print(json.dumps(result["bootstrap_difference_finetune_minus_frozen"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
