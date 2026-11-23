import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_test_data(data_path, text_col, label_col):
    if not os.path.exists(data_path):
        raise SystemExit(f"❌ Erro: O ficheiro de dados não existe em: {data_path}")
        
    data = pd.read_csv(data_path).dropna(subset=[text_col, label_col]).copy()
    data = data[~data[text_col].astype(str).str.startswith("[ERRO")]
    data = data.drop_duplicates(subset=[text_col])
    data = data[data[label_col].astype(str) != "Geral"]
    counts = data[label_col].astype(str).value_counts()
    data = data[counts[data[label_col].astype(str)].values >= 2].copy()
    
    _, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[label_col].astype(str),
    )
    return test.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Exportar previsões do conjunto de teste")
    parser.add_argument("--model", required=True, help="Caminho para a diretoria do modelo treinado")
    parser.add_argument("--data", default="data/processed/dataset_labels_normalizado.csv")
    parser.add_argument("--text", default="post_text")
    parser.add_argument("--label", default="new_sector_norm")
    parser.add_argument("--output", default="results/bertimbau_test_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    model_path = Path(args.model)
    test = load_test_data(args.data, args.text, args.label)
    
    print(f"🤖 A carregar modelo e tokenizer de: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    print(f"🚀 A extrair previsões para {len(test)} linhas de teste...")
    with torch.no_grad():
        for start in range(0, len(test), args.batch_size):
            texts = test[args.text].iloc[start:start + args.batch_size].astype(str).tolist()
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            logits = model(**tokens).logits
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    id2label = {int(key): value for key, value in model.config.id2label.items()}
    output = pd.DataFrame({
        "y_true": test[args.label].astype(str),
        "y_pred": [id2label[int(prediction)] for prediction in predictions],
    })
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output.to_csv(args.output, index=False)
    print("\n✅ Concluído com sucesso:")
    print(json.dumps({"n_test": len(output), "output": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
