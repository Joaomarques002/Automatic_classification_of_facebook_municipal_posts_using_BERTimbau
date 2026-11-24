import pandas as pd
import re
from collections import Counter

from pathlib import Path

# Raiz do projeto
ROOT_DIR = Path(__file__).resolve().parents[2]

# Dataset utilizado para análise exploratória
INPUT_FILE = (
    ROOT_DIR
    / "data"
    / "processed"
    / "dataset_labels_normalizado.csv"
)

# Coluna com o texto das publicações
TEXT_COLUMN = "post_text"
# Categoria original atribuída a cada publicação
LABEL_COLUMN = "new_sector"
# Município responsável pela publicação
MUNI_COLUMN = "municipality_name"
# Tipo de conteúdo publicado
CONTENT_TYPE_COLUMN = "content_type"

# Relatório final gerado pelo script
REPORT_FILE = (
    ROOT_DIR
    / "results"
    / "eda_report.txt"
)

# Limpeza simples para análise de frequência de palavras
def clean_text(s: str) -> str:
    # Remove URLs presentes nos textos
    s = re.sub(r"https?://\S+", " ", str(s))
    # Remove espaços duplicados
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    # Carrega o dataset para análise
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    # Número total de publicações
    total = len(df)
    # Número de publicações sem texto
    missing_text = int(df[TEXT_COLUMN].isna().sum())
    # Número de erros produzidos durante o scraping
    error_text = int(df[TEXT_COLUMN].astype(str).str.startswith("[ERRO").sum())

    # Distribuição por município
    by_muni = df[MUNI_COLUMN].value_counts().to_dict() if MUNI_COLUMN in df.columns else {}
    # Distribuição por tipo de conteúdo
    by_type = df[CONTENT_TYPE_COLUMN].value_counts().to_dict() if CONTENT_TYPE_COLUMN in df.columns else {}
    # Distribuição por categoria temática
    by_label = df[LABEL_COLUMN].value_counts().to_dict() if LABEL_COLUMN in df.columns else {}

    # Utiliza apenas textos válidos para análise de palavrasxts
    texts = df[~df[TEXT_COLUMN].astype(str).str.startswith("[ERRO")][TEXT_COLUMN].astype(str).map(clean_text)
    # Lista simples de tokens para análise exploratória
    tokens = []
    # Limita a análise aos primeiros 5000 textos
    # para evitar tempos excessivos
    for t in texts.head(5000):  # cap
        tokens.extend([w.lower() for w in re.findall(r"\b\w+\b", t) if len(w) > 2])
    # 50 palavras mais frequentes do corpus
    top = Counter(tokens).most_common(50)

    # Construção do relatório textual final
    lines = []
    lines.append(f"Total linhas: {total}")
    lines.append(f"Texto em falta: {missing_text}")
    lines.append(f"Texto com erro extração: {error_text}")
    lines.append("\nContagem por município:")
    for k,v in by_muni.items():
        lines.append(f"- {k}: {v}")
    lines.append("\nContagem por content_type:")
    for k,v in by_type.items():
        lines.append(f"- {k}: {v}")
    lines.append("\nContagem por label (new_sector):")
    for k,v in by_label.items():
        lines.append(f"- {k}: {v}")
    lines.append("\nTop tokens (amostra até 5000 textos):")
    for tok, cnt in top:
        lines.append(f"- {tok}: {cnt}")

    # Guarda o relatório em ficheiro
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    # Confirma ao utilizador onde ficou guardado o relatório    
    print(f"EDA report gravado em {REPORT_FILE}")


if __name__ == "__main__":
    main()
