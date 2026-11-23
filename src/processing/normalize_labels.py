import pandas as pd
import unicodedata
from pathlib import Path

DATASET = Path("dataset_com_texto_extraido01.csv")
MAPPING_CSV = Path("label_mapping.csv")
OUTPUT = Path("dataset_labels_normalizado.csv")
COUNTS_OUT = Path("label_counts_after_normalization.csv")

CANONICAL_SET = {
    "Cultura",
    "Saúde",
    "Ambiente, sustentabilidade e serviços públicos de água, saneamento e resíduos urbanos",
    "Juventude, educação, ciência e tecnologia",
    "Desporto",
    "Turismo, paisagem, património e gastronomia",
    "Ação social, inclusão, igualdade, terceira idade e voluntariado",
    "Economia, desenvolvimento económico e emprego",
    "Presidente, Câmara e Assembleia",
    "Segurança, proteção civil e prevenção da violência",
    "Obras municipais",
    "Geral",
}

# Merge low-frequency canonical labels into broader ones
MERGE_MAP = {
    "Festividades e efemérides (incluindo Natal)": "Cultura",
    "Mobilidade, trânsito e estacionamento": "Obras municipais",
    "Participação cidadã (incluindo orçamento participativo)": "Presidente, Câmara e Assembleia",
}


def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


def load_mapping(path: Path):
    df = pd.read_csv(path)
    mapping = {}
    for _, row in df.iterrows():
        orig = str(row['original_label']).strip()
        canon = str(row['canonical_label']).strip()
        mapping[orig] = canon
        # case-insensitive and accent-insensitive key
        key_norm = strip_accents(orig.lower())
        mapping.setdefault(f"__norm__::{key_norm}", canon)
    return mapping


def normalize_label(label: str, mapping: dict) -> str:
    # direct mapping first
    if label in mapping:
        result = mapping[label]
    else:
        norm = strip_accents(label.lower().strip())
        key = f"__norm__::{norm}"
        if key in mapping:
            result = mapping[key]
        else:
            # fallback: if already canonical, keep; else send to 'Geral'
            result = label if label in CANONICAL_SET else "Geral"

    # apply merge map if result is an old canonical slated for merge
    return MERGE_MAP.get(result, result)


def main():
    df = pd.read_csv(DATASET, encoding="utf-8")
    if 'new_sector' not in df.columns:
        raise SystemExit("Coluna 'new_sector' em falta no dataset.")

    mapping = load_mapping(MAPPING_CSV)
    df['new_sector_norm'] = df['new_sector'].astype(str).apply(lambda x: normalize_label(x, mapping))

    # Excluir entradas com label 'Geral' e avisar
    total_before = len(df)
    removed = int((df['new_sector_norm'] == 'Geral').sum())
    if removed > 0:
        pct = (removed / total_before) * 100.0 if total_before > 0 else 0.0
        print(f"Aviso: 'Geral' detectado em {removed} entradas ({pct:.2f}%). Essas entradas serão excluídas.")
    df = df[df['new_sector_norm'] != 'Geral'].copy()

    # Exportar dataset normalizado
    df.to_csv(OUTPUT, index=False, encoding="utf-8")

    # Guardar contagens pós-normalização
    counts = df['new_sector_norm'].value_counts().rename_axis('label').reset_index(name='count')
    counts.to_csv(COUNTS_OUT, index=False, encoding="utf-8")

    print("Guardado:", OUTPUT)
    print("Contagens:", COUNTS_OUT)


if __name__ == "__main__":
    main()
