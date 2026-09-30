import csv
import json
import subprocess
import sys
import os
from pathlib import Path

def parse_kv_args(argv):
    kv = {}
    for a in argv:
        if "=" in a:
            k, v = a.split("=", 1)
            kv[k.strip().lower()] = v.strip()
    return kv

def latest_output_by_prefix(base_dir: Path, prefix: str) -> Path:
    matches = sorted(
        [p for p in base_dir.glob(f"{prefix}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"Nenhuma pasta encontrada em {base_dir} com o prefixo: {prefix}")
    return matches[0]

def run_train(modeling_dir: Path, mode_name: str, freeze_encoder: bool, common_args: dict, out_prefix: str):
    # Executa o treino apontando diretamente para o script da mesma pasta
    cmd = [
        sys.executable,
        "train_bertimbau.py",
        f"file=../../{common_args['file']}", # Recua para encontrar a pasta data/
        f"label={common_args['label']}",
        f"epochs={common_args['epochs']}",
        f"bt={common_args['bt']}",
        f"be={common_args['be']}",
        f"maxlen={common_args['maxlen']}",
        f"lr={common_args['lr']}",
        f"wd={common_args['wd']}",
        f"metric={common_args['metric']}",
        f"patience={common_args['patience']}",
        f"classweights={common_args['classweights']}",
        f"max={common_args['max']}",
        f"out=../../results/{out_prefix}", # Grava dentro de results/
        f"freeze_encoder={'true' if freeze_encoder else 'false'}",
    ]

    print(f"\n⚡ === A correr Ablação: {mode_name} ===")
    subprocess.run(cmd, cwd=str(modeling_dir), check=True)

    results_dir = modeling_dir.parent.parent / "results"
    run_dir = latest_output_by_prefix(results_dir, out_prefix)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    setup = json.loads((run_dir / "training_setup.json").read_text(encoding="utf-8"))

    return {
        "mode": mode_name,
        "run_dir": str(run_dir.resolve()),
        "freeze_encoder": setup.get("freeze_encoder"),
        "trainable_parameters": setup.get("trainable_parameters"),
        "total_parameters": setup.get("total_parameters"),
        "eval_loss": metrics.get("eval_loss"),
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_f1_macro": metrics.get("eval_f1_macro"),
        "eval_f1_weighted": metrics.get("eval_f1_weighted"),
        "eval_runtime": metrics.get("eval_runtime"),
        "epoch": metrics.get("epoch"),
    }

def write_summary(results_dir: Path, rows, out_base: str):
    json_path = results_dir / f"{out_base}_summary.json"
    csv_path = results_dir / f"{out_base}_summary.csv"
    md_path = results_dir / f"{out_base}_summary.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = ["mode", "freeze_encoder", "trainable_parameters", "total_parameters", "eval_loss", "eval_accuracy", "eval_f1_macro", "eval_f1_weighted", "eval_runtime", "epoch", "run_dir"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    frozen = next(r for r in rows if r["mode"] == "frozen_head")
    finetune = next(r for r in rows if r["mode"] == "full_finetune")

    d_acc = (finetune["eval_accuracy"] or 0) - (frozen["eval_accuracy"] or 0)
    d_f1m = (finetune["eval_f1_macro"] or 0) - (frozen["eval_f1_macro"] or 0)
    d_f1w = (finetune["eval_f1_weighted"] or 0) - (frozen["eval_f1_weighted"] or 0)

    md = [
        "# Relatório de Ablação: Frozen vs Fine-tuning Completo",
        "",
        "| Configuração | Freeze encoder | Accuracy | F1 Macro | F1 Weighted | Eval Runtime | Params Treináveis |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(f"| {r['mode']} | {r['freeze_encoder']} | {r['eval_accuracy']:.4f} | {r['eval_f1_macro']:.4f} | {r['eval_f1_weighted']:.4f} | {r['eval_runtime']:.2f} | {r['trainable_parameters']} |")
    
    md.extend([
        "",
        "## Ganho de Desempenho (Fine-Tuning Completo)",
        f"- 📈 Delta Accuracy: {d_acc:+.4f}",
        f"- 📈 Delta F1 Macro: {d_f1m:+.4f}",
        f"- 📈 Delta F1 Weighted: {d_f1w:+.4f}",
        "",
        "## Diretoria dos Modelos Salvos",
    ])
    for r in rows:
        md.append(f"- **{r['mode']}**: `{r['run_dir']}`")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n📊 Relatórios de ablação gerados com sucesso na pasta results/!")

def main():
    kv = parse_kv_args(sys.argv[1:])
    modeling_dir = Path(__file__).resolve().parent
    results_dir = modeling_dir.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    common_args = {
        "file": kv.get("file", "data/dataset_labels_normalizado.csv"),
        "label": kv.get("label", "new_sector_norm"),
        "epochs": kv.get("epochs", "6"),
        "bt": kv.get("bt", "8"),
        "be": kv.get("be", "8"),
        "maxlen": kv.get("maxlen", "256"),
        "lr": kv.get("lr", "2e-5"),
        "wd": kv.get("wd", "0.01"),
        "metric": kv.get("metric", "f1_macro"),
        "patience": kv.get("patience", "2"),
        "classweights": kv.get("classweights", "true"),
        "max": kv.get("max", "0"),
    }

    out_base = kv.get("outbase", "bertimbau_ablation")
    rows = []
    
    # Executa os dois modos consecutivos
    rows.append(run_train(modeling_dir, "frozen_head", True, common_args, f"{out_base}_frozen"))
    rows.append(run_train(modeling_dir, "full_finetune", False, common_args, f"{out_base}_finetune"))

    write_summary(results_dir, rows, out_base)

if __name__ == "__main__":
    main()
