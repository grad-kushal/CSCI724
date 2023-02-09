from flask import Flask, render_template, request
from interface import get_coordinates, get_postal_code, get_breweries_by_postal_code, get_restaurants, \
    get_breweries_by_location, get_rating_google, get_rating_yelp

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
        return render_template('breweries.html', breweries=breweries)
    # if postal_code:
    #     breweries = get_breweries(postal_code)
    #     print(breweries)
    #     return render_template('breweries.html', breweries=breweries)
    else:
        return render_template('index.html', error=True)


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


@app.route('/reviews/<brewery_id>')
def reviews(brewery_id):
    """
    Render the reviews page
    :param brewery_id:
    :return:
    """
    yelp_rating = get_rating_yelp(brewery_id)
    google_rating, place_id = get_rating_google(brewery_id)
    print(yelp_rating, google_rating)
    if yelp_rating and google_rating:
        return render_template('ratings.html', yelp_rating=yelp_rating, google_rating=google_rating)
    elif yelp_rating:
        return render_template('ratings.html', yelp_rating=yelp_rating, google_rating=None)
    elif google_rating:
        return render_template('ratings.html', yelp_rating=None, google_rating=google_rating)
    else:
        return render_template('index.html', error=True)
