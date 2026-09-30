import os
import sys
import json
import random
from datetime import datetime
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import torch
from torch.utils.data import Dataset
import torch.nn as nn

# -------------------------------------------------------------------------
# CONFIGURAÇÕES E CAMINHOS PROFISSIONAIS
# -------------------------------------------------------------------------
INPUT_FILE = "data/dataset_labels_normalizado.csv"  # Aponta para a nova pasta data/
TEXT_COLUMN = "post_text"
LABEL_COLUMN = "new_sector_norm"
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"  # BERTimbau

# Centralização dos outputs na pasta results/
OUTPUT_DIR = "results/bertimbau_out"
LATEST_MODEL_FILE = "results/bertimbau_latest.txt"
LATEST_INFO_FILE = "results/bertimbau_latest_info.json"

MAX_SAMPLES = None  # None = usar dataset completo
RANDOM_SEED = 42
NUM_EPOCHS = 6
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
BATCH_TRAIN = 8
BATCH_EVAL = 8
METRIC_FOR_BEST = "f1_macro"
EARLY_STOP_PATIENCE = 2
USE_CLASS_WEIGHTS = True
DEDUP_TEXTS = True
FREEZE_ENCODER = False


def build_versioned_output_dir(base_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = f"{base_dir}_{timestamp}"
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        candidate_idx = f"{candidate}_{idx}"
        if not os.path.exists(candidate_idx):
            return candidate_idx
        idx += 1


def write_latest_model_pointer(run_output_dir: str):
    os.makedirs(os.path.dirname(LATEST_MODEL_FILE), exist_ok=True)
    with open(LATEST_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(run_output_dir) + "\n")


def write_latest_model_info(run_output_dir: str, metrics: Dict):
    os.makedirs(os.path.dirname(LATEST_INFO_FILE), exist_ok=True)
    payload = {
        "model_dir": os.path.abspath(run_output_dir),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "metrics": metrics,
        "files": {
            "metrics": os.path.abspath(os.path.join(run_output_dir, "metrics.json")),
            "labels": os.path.abspath(os.path.join(run_output_dir, "labels.txt")),
            "report": os.path.abspath(
                os.path.join(run_output_dir, "classification_report.txt")
            ),
            "confusion_matrix": os.path.abspath(
                os.path.join(run_output_dir, "confusion_matrix.png")
            ),
        },
    }
    with open(LATEST_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def set_encoder_trainable(model, trainable: bool):
    for p in model.base_model.parameters():
        p.requires_grad = trainable


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@dataclass
class TextLabelDataset(Dataset):
    encodings: Dict[str, torch.Tensor]
    labels: List[int]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_subset(path: str, max_samples: int):
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN]).copy()
    df = df[~df[TEXT_COLUMN].astype(str).str.startswith("[ERRO")]

    if DEDUP_TEXTS:
        df = df.drop_duplicates(subset=[TEXT_COLUMN]).copy()

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=RANDOM_SEED)
    return df


def prepare_label_mapping(labels: pd.Series):
    unique = sorted(labels.unique())
    id2label = {i: l for i, l in enumerate(unique)}
    label2id = {l: i for i, l in id2label.items()}
    return label2id, id2label


class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weights = kwargs.pop("class_weights", None)
        self.num_labels = kwargs.pop("num_labels")
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def main():
    argv = sys.argv[1:]
    global MAX_SAMPLES, LABEL_COLUMN, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, MAX_LENGTH, BATCH_TRAIN, BATCH_EVAL, METRIC_FOR_BEST, EARLY_STOP_PATIENCE, USE_CLASS_WEIGHTS, INPUT_FILE, OUTPUT_DIR, FREEZE_ENCODER

    for a in argv:
        if a.startswith("max="):
            try:
                _v = int(a.split("=", 1)[1])
                MAX_SAMPLES = None if _v <= 0 else _v
            except Exception:
                pass
        if a.startswith("label="):
            LABEL_COLUMN = a.split("=", 1)[1].strip()
        if a.startswith("epochs="):
            try:
                NUM_EPOCHS = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("lr="):
            try:
                LEARNING_RATE = float(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("wd="):
            try:
                WEIGHT_DECAY = float(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("maxlen="):
            try:
                MAX_LENGTH = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("bt="):
            try:
                BATCH_TRAIN = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("be="):
            try:
                BATCH_EVAL = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("metric="):
            METRIC_FOR_BEST = a.split("=", 1)[1].strip()
        if a.startswith("patience="):
            try:
                EARLY_STOP_PATIENCE = int(a.split("=", 1)[1])
            except Exception:
                pass
        if a.startswith("classweights="):
            USE_CLASS_WEIGHTS = a.split("=", 1)[1].strip().lower() in (
                "1",
                "true",
                "yes",
            )
        if a.startswith("freeze_encoder=") or a.startswith("frozen="):
            FREEZE_ENCODER = a.split("=", 1)[1].strip().lower() in ("1", "true", "yes")
        if a.startswith("file="):
            INPUT_FILE = a.split("=", 1)[1].strip()
        if a.startswith("out="):
            OUTPUT_DIR = a.split("=", 1)[1].strip()

    subset_text = "ALL" if MAX_SAMPLES is None else str(MAX_SAMPLES)
    print(
        f"Usando coluna de texto '{TEXT_COLUMN}' e label '{LABEL_COLUMN}'. Subconjunto: {subset_text}."
    )
    print(
        f"Modo de treino: {'encoder congelado (head only)' if FREEZE_ENCODER else 'fine-tuning completo'}."
    )

    run_output_dir = build_versioned_output_dir(OUTPUT_DIR)
    print(f"Diretoria de saída desta execução: '{run_output_dir}'.")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not os.path.exists(INPUT_FILE):
        raise SystemExit(
            f"❌ ERRO: Ficheiro de entrada não encontrado em '{INPUT_FILE}'. Certifica-te de que moveste o dataset para a pasta 'data/'."
        )

    df = load_subset(INPUT_FILE, MAX_SAMPLES)

    if LABEL_COLUMN in df.columns:
        geral_cnt = int((df[LABEL_COLUMN].astype(str) == "Geral").sum())
        if geral_cnt > 0:
            print(
                f"Aviso: {geral_cnt} entradas com label 'Geral' encontradas. Estas serão excluídas do treino."
            )
            df = df[df[LABEL_COLUMN].astype(str) != "Geral"].copy()

    if len(df) == 0:
        raise SystemExit(
            f"Dataset vazio após filtragem. Verifique o ficheiro '{INPUT_FILE}'."
        )
    if len(df) < 100:
        print(f"Aviso: dataset muito pequeno ({len(df)}).")

    label2id, id2label = prepare_label_mapping(df[LABEL_COLUMN].astype(str))

    counts = df[LABEL_COLUMN].astype(str).value_counts()
    valid_classes = counts[counts >= 2].index
    df = df[df[LABEL_COLUMN].astype(str).isin(valid_classes)].copy()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_COLUMN].astype(str),
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_texts(texts):
        return tokenizer(
            list(texts), padding=True, truncation=True, max_length=MAX_LENGTH
        )

    train_enc = tokenize_texts(train_df[TEXT_COLUMN].astype(str))
    test_enc = tokenize_texts(test_df[TEXT_COLUMN].astype(str))
    train_labels = [label2id[s] for s in train_df[LABEL_COLUMN].astype(str)]
    test_labels = [label2id[s] for s in test_df[LABEL_COLUMN].astype(str)]

    train_ds = TextLabelDataset(train_enc, train_labels)
    test_ds = TextLabelDataset(test_enc, test_labels)

    num_labels = len(id2label)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    set_encoder_trainable(model, trainable=not FREEZE_ENCODER)
    total_params, trainable_params = count_parameters(model)
    print(f"Parâmetros treináveis: {trainable_params}/{total_params}")

    os.makedirs(run_output_dir, exist_ok=False)
    args = TrainingArguments(
        output_dir=run_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_TRAIN,
        per_device_eval_batch_size=BATCH_EVAL,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=METRIC_FOR_BEST,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import accuracy_score, f1_score

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "f1_macro": float(f1_score(labels, preds, average="macro")),
            "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        }

    class_weights_tensor = None
    if USE_CLASS_WEIGHTS:
        train_counts = train_df[LABEL_COLUMN].astype(str).value_counts()
        weights = []
        for i in range(num_labels):
            lbl = id2label[i]
            cnt = train_counts.get(lbl, 0)
            weights.append(0.0 if cnt == 0 else 1.0 / cnt)
        class_weights_tensor = torch.tensor(weights, dtype=torch.float)

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
        class_weights=class_weights_tensor,
        num_labels=num_labels,
    )
    print("A treinar BERTimbau no subconjunto…")
    trainer.train()
    print("Avaliação…")
    metrics = trainer.evaluate()
    print(metrics)

    print("Guardando modelo e tokenizer…")
    trainer.save_model(run_output_dir)
    tokenizer.save_pretrained(run_output_dir)

    with open(os.path.join(run_output_dir, "labels.txt"), "w", encoding="utf-8") as f:
        for i in range(len(id2label)):
            f.write(f"{i}\t{id2label[i]}\n")
    with open(
        os.path.join(run_output_dir, "metrics.json"), "w", encoding="utf-8"
    ) as mf:
        json.dump(metrics, mf, ensure_ascii=False, indent=2)

    setup = {
        "input_file": INPUT_FILE,
        "text_column": TEXT_COLUMN,
        "label_column": LABEL_COLUMN,
        "model_name": MODEL_NAME,
        "freeze_encoder": FREEZE_ENCODER,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_length": MAX_LENGTH,
        "batch_train": BATCH_TRAIN,
        "batch_eval": BATCH_EVAL,
        "metric_for_best": METRIC_FOR_BEST,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "use_class_weights": USE_CLASS_WEIGHTS,
        "random_seed": RANDOM_SEED,
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
    }
    with open(
        os.path.join(run_output_dir, "training_setup.json"), "w", encoding="utf-8"
    ) as sf:
        json.dump(setup, sf, ensure_ascii=False, indent=2)

    print("Gerando relatório por classe e matriz de confusão…")
    pred_output = trainer.predict(test_ds)
    preds = np.argmax(pred_output.predictions, axis=-1)
    prediction_frame = pd.DataFrame(
        {
            "y_true": test_df[LABEL_COLUMN].astype(str).to_numpy(),
            "y_pred": [id2label[int(pred)] for pred in preds],
        }
    )
    prediction_frame.to_csv(
        os.path.join(run_output_dir, "bertimbau_predictions.csv"), index=False
    )
    target_names = [id2label[i] for i in range(len(id2label))]
    report_text = classification_report(
        [label2id[s] for s in test_df[LABEL_COLUMN].astype(str)],
        preds,
        labels=list(range(num_labels)),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    with open(
        os.path.join(run_output_dir, "classification_report.txt"), "w", encoding="utf-8"
    ) as rf:
        rf.write(report_text)

    cm = confusion_matrix(
        [label2id[s] for s in test_df[LABEL_COLUMN].astype(str)],
        preds,
        labels=list(range(num_labels)),
    )
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues")
    plt.title("Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(run_output_dir, "confusion_matrix.png"))
    plt.close()

    write_latest_model_pointer(run_output_dir)
    write_latest_model_info(run_output_dir, metrics)
    print(f"Atalho atualizado em '{LATEST_MODEL_FILE}'.")
    print(f"Resumo atualizado em '{LATEST_INFO_FILE}'.")


if name == "main":
    main()
