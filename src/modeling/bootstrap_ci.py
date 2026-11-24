import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Calcula intervalos de confiança utilizando Bootstrap
def calculate_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, ci=95, seed=42):
    # Garante que os dados são arrays NumPy
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Validação básica dos dados de entrada
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true e y_pred devem ter o mesmo tamanho e não ser vazios")
    # Gerador aleatório para permitir reproduzir os resultados
    rng = np.random.default_rng(seed)
    # Cria amostras Bootstrap com reposição
    indices = rng.integers(0, len(y_true), size=(n_bootstraps, len(y_true)))
    # Calcula o Macro F1 para cada amostra Bootstrap
    f1_values = [f1_score(y_true[index], y_pred[index], average="macro", zero_division=0) for index in indices]
    # Calcula a Accuracy para cada amostra Bootstrap
    accuracy_values = [accuracy_score(y_true[index], y_pred[index]) for index in indices]
    # Determina os limites inferior e superior do IC
    lower = (100 - ci) / 2
    upper = 100 - lower
    # Devolve valor observado e respetivo intervalo de confiança
    return {
        "n_test": len(y_true),
        "macro_f1": [f1_score(y_true, y_pred, average="macro", zero_division=0),
                     np.percentile(f1_values, lower), np.percentile(f1_values, upper)],
        "accuracy": [accuracy_score(y_true, y_pred),
                      np.percentile(accuracy_values, lower), np.percentile(accuracy_values, upper)],
        "n_bootstraps": n_bootstraps,
        "ci": ci,
        "seed": seed,
    }


def main():
    # Permite escolher o ficheiro de previsões e o local do relatório final
    parser = argparse.ArgumentParser(description="Calcular Intervalos de Confiança via Bootstrap")
    parser.add_argument("--predictions", default="results/bertimbau_test_predictions.csv")
    parser.add_argument("--output", default="results/bootstrap_ci_results.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.predictions):
        raise SystemExit(f"❌ Erro: O ficheiro de previsões não foi encontrado em: {args.predictions}")

    # Carrega as previsões produzidas pelo modelo
    frame = pd.read_csv(args.predictions)
    # Calcula os intervalos de confiança das métricas principais
    result = calculate_bootstrap_ci(frame["y_true"], frame["y_pred"])
    
    output_dir = os.path.dirname(args.output)
    # Cria automaticamente a pasta de saída se necessário
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Guarda relatório completo em formato JSON    
    with open(args.output, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    
    # Apresenta um resumo estatístico no terminal    
    print(f"\n📊 Resultados Estatísticos (N_Test = {result['n_test']}):")
    print(f"  🔹 Macro F1: {result['macro_f1'][0]:.4f} [IC 95%: {result['macro_f1'][1]:.4f}, {result['macro_f1'][2]:.4f}]")
    print(f"  🔹 Accuracy: {result['accuracy'][0]:.4f} [IC 95%: {result['accuracy'][1]:.4f}, {result['accuracy'][2]:.4f}]")
    print(f"\n✅ Relatório completo guardado em: '{args.output}'")


if __name__ == "__main__":
    main()
