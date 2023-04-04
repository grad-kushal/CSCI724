import re

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


def get_documents(mydb, collection_name, updated_year, category, rating, rating_comparison, tags, protocols):
    """
    Get the documents from the database
    :param rating_comparison: rating comparison operator
    :param mydb: database object
    :param collection_name: collection name
    :param updated_year: updated year
    :param category: category
    :param rating: rating
    :param tags: tags
    :param protocols: protocols
    :return: the documents
    """
    print("Query parameters: ", collection_name, updated_year, category, rating, tags, protocols)
    # Get the collection
    collection = mydb[collection_name]
    # Create the query
    query = {}
    if updated_year and updated_year != 'all':
        pattern = '^' + str(updated_year) + '-.*'
        sub_query = {'$regex': pattern}
        query['updated'] = sub_query
    if category and category != 'all' and collection_name == 'apis':
        query['category'] = category
    if rating_comparison and rating:
        if rating_comparison == 'gt':
            query['rating'] = {'$gt': rating}
        elif rating_comparison == 'lt':
            query['rating'] = {'$lt': rating}
        elif rating_comparison == 'eq':
            query['rating'] = {'$eq': rating}
    if tags and tags != 'all':
        tags = [tag.strip() for tag in tags]
        sub_query = {'$in': tags}
        query['tags'] = sub_query
    if protocols and protocols != 'all' and collection_name == 'apis':
        protocols = protocols.split(' ')
        sub_query = {'$in': protocols}
        query['protocols'] = sub_query
    print("Query: ", query)
    # Get the documents
    documents = collection.find(query)
    # Return the documents
    return documents


def get_documents_by_keywords(mydb, collection_name, keywords):
    """
    Get the documents from the database by keywords
    :param mydb: database object
    :param collection_name: collection name
    :param keywords: keywords list
    :return: the documents matching the keywords
    """
    print("Query parameters: ", collection_name, keywords)
    # Get the collection
    collection = mydb[collection_name]
    # Create the aggregation pipeline
    pipeline = []
    if keywords:
        for keyword in keywords:
            # Create the match stage
            match_stage = {}
            keyword = keyword.strip()
            keyword = re.escape(keyword)
            keyword = keyword.replace('\ ', '.*')
            keyword = '.*' + keyword + '.*'
            sub_query = {'$regex': keyword, '$options': 'i'}
            match_stage['$or'] = [{'name': sub_query}, {'description': sub_query}, {'summary': sub_query}]
            pipeline.append({'$match': match_stage})
    print("Pipeline: ", pipeline)
    # Get the documents
    documents = collection.aggregate(pipeline)
    # Return the documents
    return documents
