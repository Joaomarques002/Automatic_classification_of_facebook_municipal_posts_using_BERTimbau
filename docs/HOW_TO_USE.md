## normalize_labels.py

### Objetivo

Normalizar e consolidar as labels do dataset original para a taxonomia final de 11 categorias utilizada na dissertação.

### Input

data/processed/dataset_com_texto_extraido01.csv

data/raw/label_mapping.csv

### Output

data/processed/dataset_labels_normalizado.csv

results/label_counts_after_normalization.csv

### O que faz

1. Carrega o dataset extraído do Facebook.
2. Converte labels antigas para a taxonomia final.
3. Corrige diferenças de escrita e acentuação.
4. Agrupa categorias pouco representadas.
5. Remove entradas classificadas como "Geral".
6. Gera o dataset final utilizado no treino dos modelos.


## merge_new_with_labels.py

### Objetivo

Associar os novos posts recolhidos às classificações realizadas pelos estudantes, criando o dataset utilizado nos estudos da segunda fase da dissertação.

### Inputs

data/raw/dataset_new.json

data/raw/DATA_classified_by_students_2026_update.csv

### Outputs

data/processed/dataset_new_merged_labeled.csv

data/processed/dataset_new_merged_labeled_only.csv

results/dataset_new_merge_qa.json

results/dataset_new_merge_qa.txt

### O que faz

1. Carrega os novos posts recolhidos.
2. Carrega as classificações atribuídas pelos estudantes.
3. Normaliza os URLs do Facebook.
4. Resolve duplicados.
5. Associa cada publicação à respetiva categoria.
6. Gera relatórios de controlo de qualidade.
7. Produz o dataset utilizado na segunda fase da investigação.

### Ordem no pipeline

dataset_new.json\
+\
DATA_classified_by_students_2026_update.csv\
                ↓\
merge_new_with_labels.py\
                ↓\
dataset_new_merged_labeled.csv\
                ↓\
dataset_new_merged_train_ready.csv\
                ↓\
run_finetuning_ablation.py\
                ↓\
train_bertimbau.py


## train_bertimbau.py

### Objetivo

Treinar um modelo BERTimbau para classificação automática de publicações municipais portuguesas.

### Inputs

Dataset principal:

data/processed/dataset_labels_normalizado.csv

ou

data/processed/dataset_new_merged_train_ready.csv

Coluna de texto:

post_text

Coluna de classificação:

new_sector_norm

ou

CategoryName

### Outputs

results/bertimbau_out_*/

Principais artefactos gerados:

- metrics.json
- labels.txt
- training_setup.json
- bertimbau_predictions.csv
- classification_report.txt
- confusion_matrix.png
- tokenizer treinado
- modelo treinado

### O que faz

1. Carrega o dataset.
2. Remove erros e duplicados.
3. Cria os conjuntos de treino e teste.
4. Tokeniza os textos utilizando BERTimbau.
5. Treina um modelo de classificação.
6. Avalia o modelo.
7. Gera relatórios e métricas.
8. Guarda o modelo final.

### Exemplos

Treino com dataset original:

python src/modeling/train_bertimbau.py

Treino com dataset expandido:

python src/modeling/train_bertimbau.py file=data/processed/dataset_new_merged_train_ready.csv label=CategoryName

Treino com encoder congelado:

python src/modeling/train_bertimbau.py frozen=true

### Ordem no pipeline

dataset_labels_normalizado.csv\
           ↓\
train_bertimbau.py\
           ↓\
modelo treinado\
           ↓\
predict_bertimbau.py

ou

dataset_new_merged_train_ready.csv\
           ↓\
train_bertimbau.py\
           ↓\
run_finetuning_ablation.py\
           ↓\
bootstrap_ablation.py



## run_tfidf_baseline.py

### Objetivo

Treinar modelos clássicos de Machine Learning para servir como baseline de comparação com BERTimbau.

### Inputs

Dataset:

data/processed/dataset_labels_normalizado.csv

ou

data/processed/dataset_new_merged_train_ready.csv

Coluna de texto:

post_text

Coluna de classificação:

new_sector_norm

ou

CategoryName

### Modelos avaliados

- LinearSVC
- Logistic Regression
- Complement Naive Bayes

### Outputs

results/baseline_out/

Ficheiros gerados:

- metrics.json
- metrics.csv
- LinearSVC_balanced.txt
- LogisticRegression_balanced.txt
- ComplementNB.txt

Além disso são gerados ficheiros com:

- y_true
- y_pred

para cada modelo.

### O que faz

1. Carrega o dataset.
2. Remove erros e duplicados.
3. Divide os dados em treino e teste.
4. Converte os textos para TF-IDF.
5. Treina três modelos clássicos.
6. Avalia o desempenho.
7. Exporta métricas e previsões.

### Exemplo

Dataset original:

python src/modeling/run_tfidf_baseline.py

Dataset expandido:

python src/modeling/run_tfidf_baseline.py --data data/processed/dataset_new_merged_train_ready.csv --label CategoryName

### Ordem no pipeline

dataset_labels_normalizado.csv\
            ↓\
run_tfidf_baseline.py\
            ↓\
baseline clássica\
            ↓\
comparação com BERTimbau

ou

dataset_new_merged_train_ready.csv\
            ↓\
run_tfidf_baseline.py\
            ↓\
comparação com os resultados finais


## run_finetuning_ablation.py

### Objetivo

Comparar duas estratégias de utilização do BERTimbau:

1. Frozen Head
2. Full Fine-Tuning

Esta experiência permite avaliar se o ajuste de todas as camadas do modelo produz ganhos relevantes de desempenho.

### Inputs

Dataset:

data/processed/dataset_labels_normalizado.csv

ou

data/processed/dataset_new_merged_train_ready.csv

Script utilizado:

train_bertimbau.py

### Outputs

results/

São gerados:

- *_summary.json
- *_summary.csv
- *_summary.md

Além disso são criados dois modelos treinados:

- frozen_head
- full_finetune

### O que faz

1. Executa um treino com encoder congelado.
2. Executa um treino com fine-tuning completo.
3. Recolhe métricas das duas execuções.
4. Calcula diferenças de desempenho.
5. Gera relatórios automáticos.
6. Produz a base para a análise estatística posterior.

### Exemplo

Dataset original:

python src/modeling/run_finetuning_ablation.py

Dataset expandido:

python src/modeling/run_finetuning_ablation.py file=data/processed/dataset_new_merged_train_ready.csv label=CategoryName

### Ordem no pipeline

dataset_new_merged_train_ready.csv\
              ↓\
run_finetuning_ablation.py\
              ↓\
Frozen Encoder\
     vs\
Full Fine-Tuning\
              ↓\
Relatórios de comparação\
              ↓\
bootstrap_ablation.py



## predict_bertimbau.py

### Objetivo

Utilizar um modelo BERTimbau treinado para classificar automaticamente novas publicações municipais.

### Inputs

Modelo treinado:

results/bertimbau_out_*/

ou

results/bertimbau_latest.txt

Dados de entrada:

- texto individual
ou
- ficheiro CSV contendo a coluna post_text

### Outputs

CSV contendo uma nova coluna:

pred_label

com a categoria prevista para cada publicação.

### O que faz

1. Carrega um modelo previamente treinado.
2. Carrega o tokenizer associado.
3. Processa os textos.
4. Executa a classificação.
5. Produz a categoria prevista.
6. Exporta os resultados.

### Exemplo 1 - Texto único

python src/modeling/predict_bertimbau.py text="O município inaugurou um novo centro de saúde."

### Exemplo 2 - CSV completo

python src/modeling/predict_bertimbau.py csv=data/novos_posts.csv out=results/predicoes.csv

### Resultado

Input:

"O município inaugurou um novo centro de saúde."

Output:

Saúde

### Ordem no pipeline

Modelo treinado\
       ↓\
predict_bertimbau.py\
       ↓\
Novas publicações classificadas\
       ↓\
Investigação futura




## export_bertimbau_predictions.py

### Objetivo

Exportar as previsões realizadas por um modelo BERTimbau no conjunto de teste.

### Inputs

Modelo treinado:

results/bertimbau_out_*/

Dataset:

data/processed/dataset_labels_normalizado.csv

ou

data/processed/dataset_new_merged_train_ready.csv

### Outputs

results/bertimbau_test_predictions.csv

Colunas geradas:

- y_true
- y_pred

### O que faz

1. Carrega um modelo treinado.
2. Reconstrói o conjunto de teste.
3. Executa previsões.
4. Exporta classificação real e prevista.
5. Produz um ficheiro preparado para análises posteriores.

### Exemplo

python src/modeling/export_bertimbau_predictions.py --model results/bertimbau_out_20261124_120000

### Utilização típica

Este script é normalmente utilizado antes de:

- bootstrap_ci.py
- análises de erros
- matrizes de confusão
- validação estatística

### Ordem no pipeline

train_bertimbau.py\
         ↓\
modelo treinado\
         ↓\
export_bertimbau_predictions.py\
         ↓\
bertimbau_test_predictions.csv\
         ↓\
bootstrap_ci.py



## bootstrap_ci.py

### Objetivo

Calcular intervalos de confiança para as métricas do modelo utilizando Bootstrap.

### Inputs

results/bertimbau_test_predictions.csv

Este ficheiro deve conter:

- y_true
- y_pred

### Outputs

results/bootstrap_ci_results.json

### O que faz

1. Lê as previsões geradas pelo modelo.
2. Cria múltiplas amostras Bootstrap.
3. Calcula Accuracy e Macro F1 para cada amostra.
4. Estima intervalos de confiança de 95%.
5. Exporta os resultados para JSON.

### Exemplo

python src/modeling/bootstrap_ci.py

ou

python src/modeling/bootstrap_ci.py --predictions results/bertimbau_test_predictions.csv

### Resultado

O relatório inclui:

- Accuracy observada
- IC 95% da Accuracy
- Macro F1 observado
- IC 95% do Macro F1
- Número de amostras Bootstrap

### Utilização típica

Este script é utilizado após:

export_bertimbau_predictions.py

para avaliar a robustez estatística do modelo.

### Ordem no pipeline

train_bertimbau.py\
         ↓\
export_bertimbau_predictions.py\
         ↓\
bertimbau_test_predictions.csv\
         ↓\
bootstrap_ci.py\
         ↓\
bootstrap_ci_results.json


## bootstrap_ablation.py

### Objetivo

Avaliar estatisticamente as diferenças entre:

- Frozen Encoder
- Full Fine-Tuning

utilizando Bootstrap.

### Inputs

Modelos treinados produzidos por:

run_finetuning_ablation.py

Dataset:

- dataset_labels_normalizado.csv
ou
- dataset_new_merged_train_ready.csv

### Outputs

results/bootstrap_results/

Ficheiros produzidos:

- pilot_bootstrap.json
- expanded_bootstrap.json
- pilot_predictions.csv
- expanded_predictions.csv

### O que faz

1. Reconstrói o conjunto de teste.
2. Carrega os dois modelos comparados na ablação.
3. Gera previsões para ambos.
4. Executa milhares de amostragens Bootstrap.
5. Calcula diferenças entre modelos.
6. Estima intervalos de confiança.
7. Calcula a probabilidade de o ganho observado ser nulo.

### Métricas avaliadas

- Accuracy
- Macro F1
- Weighted F1

### Exemplo

Estudo piloto:

python src/modeling/bootstrap_ablation.py study=pilot

Estudo expandido:

python src/modeling/bootstrap_ablation.py study=expanded

10000 repetições Bootstrap:

python src/modeling/bootstrap_ablation.py study=expanded n=10000

### Interpretação

Se:

p_bootstrap_difference_le_zero

for muito próximo de 0,

então existe forte evidência de que o Fine-Tuning completo é superior ao modelo Frozen.

### Ordem no pipeline

dataset_new_merged_train_ready.csv\
                ↓\
run_finetuning_ablation.py\
                ↓\
Frozen Encoder\
      vs\
Full Fine-Tuning\
                ↓\
bootstrap_ablation.py\
                ↓\
Validação estatística\
                ↓\
Conclusões da dissertação


## report_bertimbau.py

### Objetivo

Inspecionar rapidamente um modelo treinado e verificar os seus principais artefactos e métricas.

### Inputs

Modelo treinado:

results/bertimbau_out_*/

ou

results/bertimbau_latest.txt

### Outputs

Não gera novos ficheiros.

Apresenta informação diretamente no terminal.

### O que faz

1. Localiza um modelo treinado.
2. Lê as métricas produzidas durante o treino.
3. Verifica a existência dos principais artefactos.
4. Conta o número de categorias utilizadas.
5. Mostra um excerto do classification report.

### Artefactos verificados

- metrics.json
- classification_report.txt
- labels.txt
- confusion_matrix.png

### Exemplo

Último modelo treinado:

```bash
python src/evaluation/report_bertimbau.py
```
### Ordem no pipeline

train_bertimbau.py\
         ↓\
modelo treinado\
         ↓\
report_bertimbau.py\
         ↓\
inspeção rápida dos resultados


## eda_report.py

### Objetivo

Realizar uma análise exploratória básica do dataset antes das etapas de normalização e treino.

### Inputs

data/processed/dataset_labels_normalizado.csv

### Outputs

results/eda_report.txt

### O que faz

1. Carrega o dataset extraído.
2. Conta publicações totais.
3. Identifica textos em falta.
4. Identifica erros de extração.
5. Analisa distribuição por município.
6. Analisa distribuição por tipo de conteúdo.
7. Analisa distribuição por categoria.
8. Calcula as palavras mais frequentes do corpus.
9. Gera um relatório textual.

### Exemplo

```bash
python src/evaluation/eda_report.py
```

### Ordem no pipeline

dataset_com_texto_extraido01.csv\
         ↓\
eda_report.py\
         ↓\
eda_report.txt\
         ↓\
normalize_labels.py