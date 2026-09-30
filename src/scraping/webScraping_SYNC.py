import os
import sys
import time
import re
import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError
from tqdm import tqdm

# -------------------------------------------------------------------------
# CONFIGURAÇÕES PRINCIPAIS
# -------------------------------------------------------------------------
STORAGE_STATE_FILE = "facebook_session.json"
INPUT_FILE = "data/DATA_socialmedia_localgov_pt.csv"  # Aponta para a nova pasta data/
OUTPUT_FILE = "data/dataset_com_texto_extraido01.csv"
LINK_COLUMN = "link"
TEXT_COLUMN = "post_text"
ROW_ID_COLUMN = "_row_id"

# Seletores do Facebook atualizados e limpos
TEXT_SELECTORS = [
    'div[data-ad-preview="message"] span[dir="auto"]',
    '[role="article"] div[dir="auto"]',
    'div[data-testid="post_message"]',
    'div[role="main"] div[dir="auto"]',
    'div[role="article"] p',
]

UI_NOISE_PATTERNS = [
    "gosto",
    "comentar",
    "partilhar",
    "escreve um comentário",
    "mais relevantes",
    "ver original",
    "ver publicação",
    "see more",
    "ver mais",
]


def get_pending_rows(df_input):
    """Verifica o ficheiro de saída e devolve apenas as linhas que ainda faltam processar."""
    if os.path.exists(OUTPUT_FILE):
        try:
            df_done = pd.read_csv(OUTPUT_FILE, encoding="utf-8", dtype=str)
            if ROW_ID_COLUMN in df_done.columns:
                done_ids = set(df_done[ROW_ID_COLUMN].dropna().astype(str))
                return df_input[~df_input.index.astype(str).isin(done_ids)].copy()
        except Exception as e:
            print(
                f"[AVISO] Falha ao ler ficheiro de progresso existente ({e}). A recomeçar do zero."
            )
    return df_input.copy()


def clean_extracted_text(text_list):
    """Remove blocos de texto que pertencem à interface (UI) do Facebook e junta o resto."""
    cleaned_blocks = []
    for block in text_list:
        text_clean = (block or "").strip()
        if len(text_clean) <= 2:
            continue
        # Ignora se o bloco contiver palavras-chave de botões da interface
        if any(noise in text_clean.lower() for noise in UI_NOISE_PATTERNS):
            continue
        cleaned_blocks.append(text_clean)

    final_text = " ".join(cleaned_blocks)
    # Limpa links residuais e espaços duplos
    final_text = re.sub(r"https?://\S+", "", final_text)
    return re.sub(r"\s+", " ", final_text).strip()


def extract_post_text(page, url):
    """Navega para o URL do Facebook e extrai o texto da publicação."""
    if not isinstance(url, str) or not url.startswith("http"):
        return "[ERRO: LINK INVÁLIDO]"

    if any(pattern in url for pattern in [".jpg", ".png", ".jpeg", "://facebook.com"]):
        return "[ERRO: LINK NÃO SUPORTADO]"

    try:
        page.goto(url, timeout=60000, wait_until="domcontentloaded")
        time.sleep(2)  # Pequena pausa para garantir renderização dinâmica

        # Tenta expandir botões de 'Ver mais' para capturar texto completo
        for button_sel in [
            'div[role="button"]:has-text("Ver mais")',
            'span[role="button"]:has-text("Ver mais")',
        ]:
            try:
                locator = page.locator(button_sel).first
                if locator.is_visible(timeout=1000):
                    locator.click(timeout=1000)
            except Exception:
                pass

        # Tenta extrair usando os melhores seletores de texto
        for selector in TEXT_SELECTORS:
            try:
                elements = page.locator(selector).all_text_contents()
                if elements:
                    cleaned = clean_extracted_text(elements)
                    if len(cleaned) > 10:
                        return cleaned
            except Exception:
                continue

        return "[ERRO: TEXTO NÃO ENCONTRADO]"
    except TimeoutError:
        return "[ERRO: TIMEOUT - O post não carregou]"
    except Exception as e:
        return f"[ERRO DESCONHECIDO: {e}]"


def main():
    if not os.path.exists(STORAGE_STATE_FILE):
        print(f"❌ ERRO: Ficheiro de sessão '{STORAGE_STATE_FILE}' não encontrado!")
        print("Execute primeiro: python src/scraping/login_facebook.py")
        return

    print(f"A carregar base de dados: {INPUT_FILE}")
    try:
        df_all = pd.read_csv(INPUT_FILE, encoding="utf-8", delimiter=";")
    except Exception:
        df_all = pd.read_csv(INPUT_FILE, encoding="latin1", delimiter=";")

    df_all[ROW_ID_COLUMN] = df_all.index.astype(str)
    df_pending = get_pending_rows(df_all)

    if len(df_pending) == 0:
        print("✅ Tudo processado! Não existem posts pendentes.")
        return

    # Definir tamanho do lote por argumento de linha de comando ou padrão (100)
    batch_size = (
        int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 100
    )
    df_lote = df_pending.head(batch_size).copy()

    print(
        f"Total pendente: {len(df_pending)} | A processar lote atual de: {len(df_lote)} posts."
    )

    with sync_playwright() as p:
        browser = p.firefox.launch(
            headless=False
        )  # Firefox é mais estável contra deteção de bots no Facebook
        context = browser.new_context(storage_state=STORAGE_STATE_FILE)
        page = context.new_page()

        results = []
        try:
            for _, row in tqdm(
                df_lote.iterrows(), total=len(df_lote), desc="Progresso do Lote"
            ):
                post_text = extract_post_text(page, row[LINK_COLUMN])
                results.append(post_text)
                time.sleep(2)  # Delay de cortesia para evitar bloqueios do Facebook
        except KeyboardInterrupt:
            print(
                "\n⏸ Execução interrompida pelo utilizador. A guardar o progresso atual..."
            )
            df_lote = df_lote.head(len(results))

        df_lote[TEXT_COLUMN] = results

        # Salvar dados de forma incremental
        if os.path.exists(OUTPUT_FILE):
            df_existing = pd.read_csv(OUTPUT_FILE, encoding="utf-8", dtype=str)
            df_final = pd.concat([df_existing, df_lote], ignore_index=True)
            df_final.drop_duplicates(subset=[ROW_ID_COLUMN], keep="last", inplace=True)
        else:
            df_final = df_lote

        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"✅ Progresso guardado com sucesso em: {OUTPUT_FILE}")
        browser.close()


if __name__ == "__main__":
    main()
