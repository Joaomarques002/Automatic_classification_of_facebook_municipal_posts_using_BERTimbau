import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Reconstrói o mesmo conjunto de teste utilizado no treino
def load_test_data(data_path, text_col, label_col):
    if not os.path.exists(data_path):
        raise SystemExit(f"❌ Erro: O ficheiro de dados não existe em: {data_path}")
    
    # Remove exemplos incompletos    
    data = pd.read_csv(data_path).dropna(subset=[text_col, label_col]).copy()
    # Remove erros produzidos durante o scraping
    data = data[~data[text_col].astype(str).str.startswith("[ERRO")]
    # Remove textos duplicados
    data = data.drop_duplicates(subset=[text_col])
    # Remove categoria genérica não utilizada no treino
    data = data[data[label_col].astype(str) != "Geral"]
    counts = data[label_col].astype(str).value_counts()
    data = data[counts[data[label_col].astype(str)].values >= 2].copy()
    
    # Repete exatamente a mesma divisão treino/teste
    # usada durante o treino dos modelos
    _, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data[label_col].astype(str),
    )
    return test.reset_index(drop=True)


# Permite escolher modelo, dataset e ficheiro de saída
# através da linha de comandos
def main():
    parser = argparse.ArgumentParser(description="Exportar previsões do conjunto de teste")
    parser.add_argument("--model", required=True, help="Caminho para a diretoria do modelo treinado")
    parser.add_argument("--data", default="data/processed/dataset_labels_normalizado.csv")
    parser.add_argument("--text", default="post_text")
    parser.add_argument("--label", default="new_sector_norm")
    parser.add_argument("--output", default="results/bertimbau_test_predictions.csv")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Carrega o modelo previamente treinado
    model_path = Path(args.model)
    test = load_test_data(args.data, args.text, args.label)
    
    print(f"🤖 A carregar modelo e tokenizer de: {model_path}")
    # Carrega tokenizer e pesos do modelo
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    # Utiliza GPU se estiver disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    predictions = []
    print(f"🚀 A extrair previsões para {len(test)} linhas de teste...")
    # Gera previsões para o conjunto de teste
    with torch.no_grad():
        # Processamento em lotes para reduzir consumo de memória
        for start in range(0, len(test), args.batch_size):
            texts = test[args.text].iloc[start:start + args.batch_size].astype(str).tolist()
            # Conversão dos textos para o formato esperado pelo modelo
            tokens = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokens = {key: value.to(device) for key, value in tokens.items()}
            logits = model(**tokens).logits
            # Classe prevista para cada publicação
            predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())

    # Converte IDs numéricos novamente para nomes das categorias
    id2label = {int(key): value for key, value in model.config.id2label.items()}
    # Exporta classificação real e classificação prevista
    output = pd.DataFrame({
        "y_true": test[args.label].astype(str),
        "y_pred": [id2label[int(prediction)] for prediction in predictions],
    })
    
    
    output_dir = os.path.dirname(args.output)
    if output_dir:
        # Cria automaticamente a pasta de saída caso não exista
        os.makedirs(output_dir, exist_ok=True)
    
    # Guarda o ficheiro utilizado em análises posteriores
    output.to_csv(args.output, index=False)
    print("\n✅ Concluído com sucesso:")
    print(json.dumps({"n_test": len(output), "output": str(Path(args.output).resolve())}, indent=2))


if __name__ == "__main__":
    main()
