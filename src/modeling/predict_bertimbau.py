import os
import sys
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Localização do último modelo treinado
DEFAULT_MODEL_DIR = "results/bertimbau_out"
# Ficheiro que aponta para o modelo mais recente
LATEST_MODEL_FILE = "results/bertimbau_latest.txt"
# Coluna que contém o texto das publicações
TEXT_COLUMN = "post_text"
# Coluna que irá receber a previsão do modelo
LABEL_COLUMN = "pred_label"
BATCH_SIZE = 32  # Otimização crucial para evitar quebras de memória


# Carrega o mapeamento entre IDs numéricos
# e os nomes das categorias
def load_labels(model_dir):
    path = os.path.join(model_dir, "labels.txt")
    id2label = {}
    if not os.path.exists(path):
        raise SystemExit(f"❌ Erro: Ficheiro de mapeamento de etiquetas não encontrado em: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            i, name = line.split("\t", 1)
            id2label[int(i)] = name
    return id2label


# Determina qual o modelo a utilizar
# (especificado manualmente ou último treino disponível)
def resolve_model_dir(cli_model_dir: str = None):
    # Caso o utilizador forneça explicitamente um modelo
    if cli_model_dir:
        if os.path.isdir(cli_model_dir):
            return cli_model_dir
        # Tenta procurar dentro de results caso passem apenas o nome da pasta
        alternative = os.path.join("results", cli_model_dir)
        if os.path.isdir(alternative):
            return alternative
        raise SystemExit(f"❌ Erro: A diretoria do modelo especificada não existe: {cli_model_dir}")

    # Tenta utilizar automaticamente o último modelo treinado
    if os.path.exists(LATEST_MODEL_FILE):
        with open(LATEST_MODEL_FILE, "r", encoding="utf-8") as f:
            candidate = f.read().strip().lstrip("\ufeff")
        if candidate and os.path.isdir(candidate):
            return candidate

    if os.path.isdir(DEFAULT_MODEL_DIR):
        return DEFAULT_MODEL_DIR

    raise SystemExit(
        f"❌ Erro: Modelo não encontrado. Execute primeiro o treino ou especifique o caminho manualmente usando: model=CAMINHO"
    )

# Carrega tokenizer e modelo treinado
def predict_texts(texts, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    
    # Utiliza GPU caso esteja disponível
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    id2label = load_labels(model_dir)
    all_preds = []
    
    # Processamento em lotes para reduzir consumo de memória
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="A classificar posts"):
        batch_texts = texts[i : i + BATCH_SIZE]
        
        # Converte texto para o formato esperado pelo BERTimbau
        enc = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        
        # Mover dados para o mesmo dispositivo do modelo
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = model(**enc).logits
            # Obtém a classe mais provável para cada publicação
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
        all_preds.extend([id2label[int(p)] for p in preds])
        
    return all_preds


def main():
    argv = sys.argv[1:]
    if len(argv) == 0:
        print(
            "📋 Modo de Utilização:\n"
            "  👉 Classificar uma frase única:\n"
            "     python src/modeling/predict_bertimbau.py text=\"O município aprovou as novas obras na estrada.\"\n\n"
            "  👉 Classificar um ficheiro CSV completo:\n"
            "     python src/modeling/predict_bertimbau.py csv=data/dataset_com_texto_extraido01.csv out=results/posts_classificados.csv\n\n"
            "  👉 Especificar um modelo específico:\n"
            "     python src/modeling/predict_bertimbau.py text=\"Frase teste\" model=results/bertimbau_out_20260930_120000"
        )
        sys.exit(0)
        
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip().lower()] = v.strip()

    model_dir = resolve_model_dir(kv.get("model"))
    print(f"🤖 A utilizar o modelo carregado de: {model_dir}")

    # Permite classificar uma frase diretamente no terminal
    if "text" in kv:
        preds = predict_texts([kv["text"]], model_dir)
        print(f"\n🏷️ Resultado da Classificação: {preds[0]}")
        return

    # Permite classificar milhares de publicações de uma só vez
    csv_path = kv.get("csv")
    out_path = kv.get("out", "results/bertimbau_predictions_output.csv")
    
    if not csv_path:
        print("❌ Erro: Especifique o caminho do ficheiro de dados usando csv=data/ficheiro.csv")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"❌ Erro: O ficheiro de entrada especificado não existe em: {csv_path}")
        sys.exit(1)

    print(f"📊 A ler dados de: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8")
    
    # Verifica se a coluna de texto existe
    if TEXT_COLUMN not in df.columns:
        print(f"❌ Erro: A coluna '{TEXT_COLUMN}' não foi encontrada no CSV fornecido.")
        sys.exit(1)
        
    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    
    # Remove publicações com erros de scraping
    df = df[~df[TEXT_COLUMN].str.startswith("[ERRO")].copy() 

    texts = list(df[TEXT_COLUMN])
    print(f"🚀 A processar {len(texts)} linhas com o BERTimbau...")
    
    preds = predict_texts(texts, model_dir)
    # Adiciona a previsão produzida pelo modelo
    df[LABEL_COLUMN] = preds
    
    # Garantir que a pasta de destino existe
    output_dir = os.path.dirname(out_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Guarda o resultado final num novo CSV
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✨ Concluído! Resultados guardados com sucesso em: {out_path}")


if __name__ == "__main__":
    main()
