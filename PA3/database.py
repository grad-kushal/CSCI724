import pymongo


def delete_collection(db, collection):
    """
    Delete the specified collection from the database
    :param db: database name
    :param collection: collection name
    :return: the result of the delete operation
    """
    return db[collection].drop()


def connect(connection_string="mongodb://localhost:27017/"):
    """
    Connect to the MongoDB server and return a reference to the database.
    :return: a reference to the database
    """
    print("Connecting to MongoDB...")
    client = pymongo.MongoClient(connection_string)
    return client["csci724_pa3"]


def insert_document(db, collection, data):
    """
    Insert a document into the specified collection in the database
    :param db: database name
    :param collection: collection name
    :param data: data to insert
    :return: the result of the insert operation
    """
    return db[collection].insert_one(data)


def insert_documents(db, collection, data):
    """
    Insert multiple documents into the specified collection in the database
    :param db: database name
    :param collection: collection name
    :param data: data to insert
    :return: the result of the insert operation
    """
    return db[collection].insert_many(data)


def main():
    mydb = connect()

    mashups_collection = "mashups"
    insert_document(mydb, mashups_collection, {})


if __name__ == '__main__':
    main()


def get_documents(mydb, collection_name, updated_year, category, rating, tags):
    """
    Get the documents from the database
    :param mydb: database object
    :param collection_name: collection name
    :param updated_year: updated year
    :param category: category
    :param rating: rating
    :param tags: tags
    :return: the documents
    """
    print("Query parameters: ", collection_name, updated_year, category, rating, tags)
    # Get the collection
    collection = mydb[collection_name]
    # Create the query
    query = {}
    if updated_year and updated_year != 'all':
        query['updated_year'] =
    if category and category != 'all':
        query['category'] = category
    if rating and rating != 'all':
        query['rating'] = str(rating)
    if tags and tags != 'all':
        print("Tags: ", tags)
        tags_new = ",".join(tags)
        tags_param = "[" + tags_new + "]"
        query['tags'] = {'$all': tags}
    print("Query: ", query)
    # Get the documents
    documents = collection.find(query)
    # Return the documents
    return documents
