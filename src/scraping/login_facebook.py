import asyncio
import os
from playwright.async_api import async_playwright
from dotenv import load_dotenv

# Carrega as credenciais do ficheiro .env de forma segura
load_dotenv()

STORAGE_STATE_FILE = "facebook_session.json"
FACEBOOK_EMAIL = os.getenv("FB_EMAIL")
FACEBOOK_PASSWORD = os.getenv("FB_PASSWORD")

async def login():
    if not FACEBOOK_EMAIL or not FACEBOOK_PASSWORD:
        print("❌ ERRO: Credenciais não encontradas no ficheiro .env!")
        return

    async with async_playwright() as p:
        print("A abrir o navegador Chromium...")
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()

        # 1. Navegar para o Facebook
        await page.goto("https://facebook.com")
        await page.wait_for_load_state("domcontentloaded")

        # 2. Preencher formulário
        await page.fill('#email', FACEBOOK_EMAIL)
        await page.fill('#pass', FACEBOOK_PASSWORD)

        # 3. Clicar no botão de login
        try:
            await page.wait_for_selector('[data-testid="royal_login_button"]', timeout=5000)
            await page.click('[data-testid="royal_login_button"]')
        except Exception as e:
            print(f"[AVISO] Botão padrão não clicado (redirecionamento ou mudança de layout): {e}")

        print("\n=======================================================")
        print("🛑 AÇÃO MANUAL NECESSÁRIA! 🛑")
        print("Resolva o 2FA / Verificação de Segurança que abriu no browser.")
        print("O script vai aguardar 5 minutos (300 segundos)...")
        print("=======================================================\n")
        
        await asyncio.sleep(300) 

        print("A tentar validar o login e guardar a sessão...")
        
        # 4. Verificar se o login teve sucesso (procura por elementos comuns da Home)
        try:
            await page.wait_for_selector('[role="feed"], [aria-label="Facebook"]', timeout=30000) 
            # 5. Guardar cookies e estado da sessão
            await context.storage_state(path=STORAGE_STATE_FILE)
            print(f"\n✅ SESSÃO SALVA com sucesso em {STORAGE_STATE_FILE}!")
        except Exception as e:
            print(f"❌ ERRO: Não foi possível confirmar o login na página principal: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(login())
