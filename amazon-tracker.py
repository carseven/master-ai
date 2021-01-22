from bs4 import BeautifulSoup as bs
import requests
import re
import smtplib
import time

URL = 'https://www.amazon.es/Beyerdynamic-770-PRO-Auriculares-estudio/dp/B0016MNAAI/ref=sr_1_1?__mk_es_ES=ÅMÅŽÕÑ&crid=3RZLQUARRBLXP&dchild=1&keywords=beyerdynamic+dt+770+pro&qid=1611327024&sprefix=beyer%2Caps%2C224&sr=8-1'

password = "biiexjwfqxdgferj"
mail = "carles.serra33@gmail.com"
buy_price = 100
resquest_time_interval = 60 * 60 * 24


def check_amazon_price(URL: str) -> float:
    """Devuelve el precio de un producto de AMAZON.es

    Args:
        URL (str): URL de AMAZON.ES

    Returns:
        float: Precio del procudto
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.2 Safari/605.1.15"
    }

    # Https url call
    try:
        page = requests.get(URL, headers=headers)
    except:
        raise Exception('La URL no existe: ' + URL)

    try:
        # Tranform https reponse to soup object (html)
        soup = bs(page.content, "html.parser")
        euro_price = soup.find(id="priceblock_ourprice").get_text().strip()

        # Obtened only the price digits and cast to float
        return float(re.search("\d+", euro_price).group())
    except:
        print('No se ha encontrado el precio, el articulo no disponible.')


def send_mail(mail, password, URL):
    server = smtplib.SMTP('smtp.gmail.com')
    server.ehlo()
    server.starttls()
    server.ehlo()

    server.login(mail, password)

    subject = 'El precio ha bajado!'
    body = u'Link de amazon '

    message = f"Subject: {subject}\n\n{body}"

    server.sendmail(
        mail,
        mail,
        message
    )


while(True):
    price = check_amazon_price(URL)
    if price is not None and price <= buy_price:
        send_mail(mail, password, URL)

    time.sleep(resquest_time_interval)
