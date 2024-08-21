from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
import time
from pathlib import Path
import requests

PIXABAY_IMG_URL = "https://cdn-au.onetrust.com/logos/static/powered_by_logo.svg"

def scroll_to_bottom(driver):
    """Desplázate hasta el final de la página con desplazamiento suave."""
    last_height = driver.execute_script("return document.body.scrollHeight - 1400")
    pos = last_height / 3
    i = 1
    while i < 3:
        # Desplázate hasta el final de la página
        driver.execute_script(f"window.scroll({{top: {pos * i}, left: 0, behavior: 'smooth'}});")
        time.sleep(3)  # Espera para que el scroll suave se complete
        # Espera a que se cargue el contenido
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'img'))
            )
            i+=1
        except Exception as e:
            print(f"Error durante la espera del contenido: {e}")
            break

def donwload_images(images:list[str]):
    folder_images = Path('./imgs')
    index = max([int(img.name.split('img')[1].split('.')[0]) for img in folder_images.iterdir()] or [0]) + 1

    for url in images:
        if url == PIXABAY_IMG_URL or url.endswith("logo.png"): continue
        r = requests.get(url, allow_redirects=False)
        file_name = f"img{index}.jpg"
        file_path = folder_images / file_name
        open(file_path, 'wb').write(r.content)
        index += 1

def process_page(driver):
    scroll_to_bottom(driver)
    images = [img.get_property('src') for img in driver.find_elements(By.TAG_NAME, 'img') if not img.get_property('src').endswith('blank.gif')]
    donwload_images(images)    

def next_page(driver):
    while True:
        """Navega a la siguiente página, devuelve True si hay una siguiente página, False en caso contrario."""
        try:
            # Encuentra y hace clic en el botón o enlace de la siguiente página
            next_button = driver.find_element(By.LINK_TEXT, 'Página siguiente')  # O usa otro selector según el sitio web
            next_button.click()
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'img'))  # Asegúrate de que haya imágenes en la nueva página
            )
            return True
        except StaleElementReferenceException:
            # El elemento se ha vuelto obsoleto, vuelve a intentar buscarlo
            print("Elemento obsoleto, reintentando...")
            continue
        except NoSuchElementException:
            # No hay un botón de siguiente página, probablemente se ha alcanzado el final
            return False
        except Exception as e:
            print(f"Error navegando a la siguiente página: {e}")
            return False

gecko_options = webdriver.FirefoxOptions()
# gecko_options.add_argument("--headless") 

driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=gecko_options)

wait = WebDriverWait(driver, 10)

url = r"https://pixabay.com/es/images/search/fondos%20de%20colores/"

driver.get(url)
wait.until(EC.url_to_be(url))
while True:
    try:
        process_page(driver)
        if not next_page(driver):
            break
    except:
        driver.quit()
