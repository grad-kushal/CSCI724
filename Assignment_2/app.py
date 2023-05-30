import math
import xml.etree.ElementTree as ET

import requests.sessions
from flask import Flask, render_template, request

from web_interface import get_restaurants, \
    get_breweries_by_location, get_rating_google, get_rating_yelp
import mysql.connector
import uuid

app = Flask(__name__)


@app.route('/')
def index():
    """
    Render the index page
    :return:
    """
    return render_template('index.html')


@app.route('/breweries', methods=['POST'])
def breweries():
    """
    Render the breweries page
    :return:
    """
    conversation_id = str(uuid.uuid4())
    address = request.form['address']
    loc = address.replace("(", "").replace(")", "").split(",")
    lat = float(loc[0].strip())
    lng = float(loc[1].strip())
    print(lat, lng)
    # lat, lng = get_coordinates(address)
    # postal_code = get_postal_code(lat, lng)
    # print(postal_code)
    if lat and lng:
        breweries = get_breweries_by_location(lat, lng)
        return render_template('breweries.html', breweries=breweries, conversation_id=conversation_id)
    else:
        return render_template('error.html', error=True)


@app.route('/restaurants', methods=['POST'])
def restaurants():
    """
    Render the restaurants page
    :return:
    """
    address = request.form['address']
    loc = address.replace("(", "").replace(")", "").split(",")
    lat = float(loc[0].strip())
    lng = float(loc[1].strip())
    print(lat, lng)
    # lat, lng = get_coordinates(address)
    restaurants = get_restaurants(lat, lng, "chinese", 2)
    print(restaurants)
    return render_template('restaurants.html', restaurants=restaurants)


@app.errorhandler(404)
def page_not_found(e):
    """
    Render the error page
    :param e:
    :return:
    """
    return render_template('error.html', error=True)


@app.route('/reviews/<conversation_id>/<brewery_id>')
def reviews(brewery_id, conversation_id):
    """
    Render the reviews page
    :param brewery_id:  the id of the brewery
    :param conversation_id: the id of the conversation
    :return: ratings
    """
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="pa2"
    )
    print(mydb, "KKKKKKKKKKKKKKKKK")
    cur = mydb.cursor()
    if conversation_id is not None:
        cur.execute(f"SELECT * FROM conversation WHERE conversation_id = '{conversation_id}'")
        result = cur.fetchone()
        if result is None:
            cur.execute(f"INSERT INTO conversation VALUES ('{conversation_id}', '{brewery_id}')")
            mydb.commit()
        else:
            cur.execute(f"UPDATE conversation SET brewery_id = '{brewery_id}' WHERE conversation_id = '{conversation_id}'")
            mydb.commit()
    cur.execute(f"SELECT * FROM brewery WHERE id = '{brewery_id}'")
    result = cur.fetchone()
    if result is not None:
        yelp_rating, google_rating = result[2], result[1]
    else:
        yelp_rating = get_rating_yelp(brewery_id)
        google_rating = get_rating_google(brewery_id)
        brewery = {
            "id": brewery_id,
            "yelp": yelp_rating if yelp_rating else 0.0,
            "google": google_rating if google_rating else 0.0
        }
        cur.execute(f"INSERT INTO brewery VALUES ('{brewery['id']}', '{brewery['google']}', '{brewery['yelp']}')")
        mydb.commit()
    if yelp_rating:
        yelp_rating_floor = math.trunc(yelp_rating)
        print("Yelp rating trunc: ", yelp_rating_floor)
        soap_url = "https://number-conversion-service.p.rapidapi.com/webservicesserver/NumberConversion.wso"
        request_body = f"<?xml version='1.0' encoding='utf-8'?><soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/'><soap:Body><NumberToWords xmlns='http://www.dataaccess.com/webservicesserver/'><ubiNum>{yelp_rating_floor}</ubiNum></NumberToWords></soap:Body></soap:Envelope>"
        headers = {
            "content-type": "application/xml",
            "X-RapidAPI-Key": "",
            "X-RapidAPI-Host": "number-conversion-service.p.rapidapi.com"
        }
        response = requests.request("POST", soap_url, data=request_body, headers=headers)
        root = ET.fromstring(response.text)
        yelp_word = root[0][0][0].text + " star"
        print("Yelp word: ", yelp_word)
    if google_rating:
        google_rating_floor = math.trunc(google_rating)
        print("Google rating trunc: ", google_rating_floor)
        soap_url = "https://number-conversion-service.p.rapidapi.com/webservicesserver/NumberConversion.wso"
        request_body = f"<?xml version='1.0' encoding='utf-8'?><soap:Envelope xmlns:soap='http://schemas.xmlsoap.org/soap/envelope/'><soap:Body><NumberToWords xmlns='http://www.dataaccess.com/webservicesserver/'><ubiNum>{google_rating_floor}</ubiNum></NumberToWords></soap:Body></soap:Envelope>"
        headers = {
            "content-type": "application/xml",
            "X-RapidAPI-Key": "c274a97723msh69e78268b2374a8p14b8c3jsn4058e62a4501",
            "X-RapidAPI-Host": "number-conversion-service.p.rapidapi.com"
        }
        response = requests.request("POST", soap_url, data=request_body, headers=headers)
        root = ET.fromstring(response.text)
        google_word = root[0][0][0].text + " star"
        print("Google word: ", google_word)
    print(yelp_rating, google_rating)
    if yelp_rating and google_rating:
        return render_template('ratings.html', yelp_rating=yelp_rating, google_rating=google_rating,
                               yelp_word=yelp_word, google_word=google_word)
    elif yelp_rating:
        return render_template('ratings.html', yelp_rating=yelp_rating, google_rating=None, yelp_word=yelp_word,
                               google_word=None)
    elif google_rating:
        return render_template('ratings.html', yelp_rating=None, google_rating=google_rating, yelp_word=None,
                               google_word=google_word)
    else:
        return render_template('error.html', error=True)
