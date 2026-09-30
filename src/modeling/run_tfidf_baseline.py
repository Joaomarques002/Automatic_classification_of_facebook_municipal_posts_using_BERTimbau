import argparse
import json
import time
import os
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

def load_data(path, text_col, label_col):
    if not os.path.exists(path):
        raise SystemExit(f"❌ Erro: O ficheiro de dados não existe em: {path}")
        
    data = pd.read_csv(path).dropna(subset=[text_col, label_col]).copy()
    data = data[~data[text_col].astype(str).str.startswith("[ERRO")]
    data = data.drop_duplicates(subset=[text_col])
    data = data[data[label_col].astype(str) != "Geral"]
    counts = data[label_col].astype(str).value_counts()
    return data[counts[data[label_col].astype(str)].values >= 2].copy()

def main():
    parser = argparse.ArgumentParser(description="Treino de Modelos Baseline (TF-IDF)")
    parser.add_argument("--data", default="data/dataset_labels_normalizado.csv")
    parser.add_argument("--text", default="post_text")
    parser.add_argument("--label", default="new_sector_norm")
    parser.add_argument("--output", default="results/baseline_out")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    args = parser.parse_args()

    print(f"📊 A carregar dados para o Baseline a partir de: {args.data if not args.train else args.train}")

    if args.train and args.test:
        train_df = load_data(args.train, args.text, args.label)
        test_df = load_data(args.test, args.text, args.label)
    else:
        data = load_data(args.data, args.text, args.label)
        train_df, test_df = train_test_split(
            data, test_size=0.2, random_state=42,
            stratify=data[args.label].astype(str),
        )

    # Vetorização de Texto
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True, min_df=2)
    x_train = vectorizer.fit_transform(train_df[args.text].astype(str))
    x_test = vectorizer.transform(test_df[args.text].astype(str))
    y_train = train_df[args.label].astype(str)
    y_test = test_df[args.label].astype(str)
    
    models = {
        "LinearSVC_balanced": LinearSVC(class_weight="balanced", random_state=42),
        "LogisticRegression_balanced": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "ComplementNB": ComplementNB(),
    }
    
    results = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 A treinar {len(models)} modelos clássicos...")
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        
        prediction_frame = pd.DataFrame({
            "y_true": y_test.to_numpy(),
            "y_pred": predictions,
        })
        prediction_frame.to_csv(output_dir / f"{name}_predictions.csv", index=False)
        
        metrics = {
            "model": name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
            "f1_weighted": f1_score(y_test, predictions, average="weighted"),
            "train_time_seconds": time.perf_counter() - started,
        }
        results.append(metrics)
        
        report = classification_report(y_test, predictions, digits=4, zero_division=0)
        (output_dir / f"{name}.txt").write_text(json.dumps(metrics, indent=2) + "\n\n" + report, encoding="utf-8")
        print(f"  ✨ {name}: accuracy={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")
        
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(output_dir / "metrics.csv", index=False)
    print(f"✅ Execução concluída. Resultados guardados na pasta: '{args.output}'")

if __name__ == "__main__":
    main()
