import database
from parser import read_mashup_data, read_api_data


def read_data():
    """
    Read the data from the files
    :return: None
    """
    # Read the mashup data
    mashup_data = read_mashup_data('data/mashup.txt')
    # Read the API data
    api_data = read_api_data('data/api.txt')
    # Return the data
    return mashup_data, api_data


def main():
    """
    Main function
    :return: None
    """
    # Read the data
    mashup_data, api_data = read_data()
    # Connect to the database
    mydb = database.connect('mongodb://localhost:27017/')
    # Insert the data into the database
    database.insert_documents(mydb, 'mashups', mashup_data)
    database.insert_documents(mydb, 'apis', api_data)


if __name__ == '__main__':
    app.run()

