import pandas as pd
import re
from collections import Counter

INPUT_FILE = "dataset_com_texto_extraido01.csv"
TEXT_COLUMN = "post_text"
LABEL_COLUMN = "new_sector"
MUNI_COLUMN = "municipality_name"
CONTENT_TYPE_COLUMN = "content_type"
REPORT_FILE = "eda_report.txt"


def clean_text(s: str) -> str:
    s = re.sub(r"https?://\S+", " ", str(s))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
    total = len(df)
    missing_text = int(df[TEXT_COLUMN].isna().sum())
    error_text = int(df[TEXT_COLUMN].astype(str).str.startswith("[ERRO").sum())

    by_muni = df[MUNI_COLUMN].value_counts().to_dict() if MUNI_COLUMN in df.columns else {}
    by_type = df[CONTENT_TYPE_COLUMN].value_counts().to_dict() if CONTENT_TYPE_COLUMN in df.columns else {}
    by_label = df[LABEL_COLUMN].value_counts().to_dict() if LABEL_COLUMN in df.columns else {}

    # top tokens (very rough) from non-error texts
    texts = df[~df[TEXT_COLUMN].astype(str).str.startswith("[ERRO")][TEXT_COLUMN].astype(str).map(clean_text)
    tokens = []
    for t in texts.head(5000):  # cap
        tokens.extend([w.lower() for w in re.findall(r"\b\w+\b", t) if len(w) > 2])
    top = Counter(tokens).most_common(50)

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

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"EDA report gravado em {REPORT_FILE}")


if __name__ == "__main__":
    main()
