import requests
import json


def get_rating_google(brewery_name):
    """
    Get the reviews of a brewery
    :return:
    """
    api_key = 'AIzaSyBmCq5Z5o5s0GAgYHdcuQpkDyzZhCBAwok'
    url = f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input={brewery_name}&inputtype=textquery&fields=place_id,rating&key={api_key}"
    # url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={brewery_name}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        return data['candidates'][0]['rating']


def get_rating_yelp(brewery_name):
    """
    Get the reviews of a brewery
    :param brewery_name:
    :return:
    """
    url = f"https://api.yelp.com/v3/businesses/search?term={brewery_name}&location={brewery_name}"
    headers = {
        "Authorization": "Bearer RH9s9CxW28JIigcr-oLdxpBHlkrlpvgeko3diqEtDJf42uitR-lOuVdXO9UiFhb6jGX4Ip0IjuwVaAC_9qtpwzeTnkKtuzvuSKBiiFMdlmyvtNN0uy89MQ57w5zhY3Yx",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    if data['businesses']:
        return data['businesses'][0]['rating']


def get_coordinates(address):
    """
    Get the latitude and longitude of an address
    :param address:
    :return:
    """
    api_key = 'AIzaSyBmCq5Z5o5s0GAgYHdcuQpkDyzZhCBAwok'
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"

    response = requests.get(geocode_url)
    data = response.json()
    if data['status'] == 'OK':
        location = data['results'][0]['geometry']['location']
        return location['lat'], location['lng']
    return None, None


def get_restaurants(lat, lng, cuisine, radius):
    """
    Get the restaurants located near a location
    :param radius: distance in miles
    :param cuisine: cuisine type
    :param lat: latitude
    :param lng: longitude
    :return: list of restaurants
    """
    spoonacular_api_key = "85e6d9a39993484b9be959611f31e179"
    spoonacular_url = f"https://api.spoonacular.com/food/restaurants/search?distance={radius}&cuisine={cuisine}&latitude={lat}&longitude={lng}&apiKey={spoonacular_api_key}"
    response = requests.get(spoonacular_url)
    return response.json()


def get_breweries_by_postal_code(postal_code):
    """
    Get the breweries located in a postal code
    :param postal_code:
    :return: list of breweries
    """
    openbrewery_url = f"https://api.openbrewerydb.org/breweries?by_postal={postal_code}"
    response = requests.get(openbrewery_url)
    return response.json()


def get_breweries_by_location(lat, lng):
    """
    Get the breweries located near a location
    :param lat: latitude
    :param lng: longitude
    :return:    list of breweries
    """
    openbrewery_url = f"https://api.openbrewerydb.org/breweries?by_dist={lat},{lng}"
    response = requests.get(openbrewery_url)
    data = response.json()
    print(data)
    return data


def get_postal_code(lat, lng):
    """
    Get the postal code of a location
    :param lat: latitude
    :param lng: longitude
    :return:    postal code
    """
    api_key = "bd1c7a27bdf64214b415d1c1475b680f"
    reverse_geocode_url = f"https://api.opencagedata.com/geocode/v1/json?q={lat}+{lng}&key={api_key}"
    response = requests.get(reverse_geocode_url)
    data = response.json()
    result = data['results']
    if result:
        components = result[0]['components']
        if 'postcode' in components:
            return components['postcode']
    return None


def main():
    address = input("Enter a location: ")
    lat, lng = get_coordinates(address)
    breweries = get_breweries_by_location(lat, lng)
    # get_restaurants(lat, lng, "chinese", 2)
    # postal_code = get_postal_code(lat, lng)
    print(get_rating_yelp("fifth-frame-brewing-co-rochester"))
    print(get_rating_google("fifth-frame-brewing-co-rochester"))
    # if postal_code:
    #     breweries = get_breweries_by_postal_code(postal_code)
    #     print(breweries)
        # print(f"Top 5 closest breweries near {address}:")
        # for i, brewery in enumerate(breweries):
        #     print(f"{i + 1}. {brewery['name']} ({brewery['street']})")
    # else:
    #     print("Invalid address.")


if __name__ == '__main__':
    main()
