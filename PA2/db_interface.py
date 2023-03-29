import mysql


def exists_in_database(brewery_id):
    """
    Check if the brewery exists in the database
    :param brewery_id: the id of the brewery
    :return: True if it exists, False otherwise
    """
    cursor = connect()
    cursor.execute(f"SELECT * FROM breweries WHERE id = '{brewery_id}'")
    return cursor.fetchone()


def insert_brewery_rating(brewery):
    """
    Insert a brewery into the database
    :param brewery: the brewery
    :return: None
    """
    cursor = connect()
    cursor.execute(f"INSERT INTO ratings VALUES ('{brewery['id']}', '{brewery['google']}', '{brewery['yelp']}'")
    cursor.commit()


def get_ratings_from_database(brewery_id):
    """
    Get the ratings of a brewery from the database
    :param brewery_id: the id of the brewery
    :return: the ratings
    """
    cursor = connect()
    cursor.execute(f"SELECT * FROM ratings WHERE id = '{brewery_id}'")
    result = cursor.fetchone()
    return result[1], result[2]


def connect():
    """
    Connect to the database
    :return: the cursor
    """
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="pa2"
    )
    return mydb.cursor()
