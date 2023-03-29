import parser

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

import database

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    """
    Render the index page
    :return:
    """
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    """
    Render the results page
    :return:
    """
    collection_name = request.form['search-type']
    updated_year = request.form['updated-year']
    category = request.form['category']
    rating = request.form['rating']
    tags = request.form['tags']
    tag_list = []
    if tags:
        tag_list = tags.split(',')
    mydb = database.connect()
    if len(mydb.list_collection_names()) != 2:
        database.delete_collection(mydb, 'mashups')
        database.delete_collection(mydb, 'apis')
        print("Obsolete database. Fixing it...")
        mashup_data, api_data = parser.read_data()
        database.insert_documents(mydb, 'mashups', mashup_data)
        database.insert_documents(mydb, 'apis', api_data)
    result = database.get_documents(mydb, collection_name, updated_year, category, rating, tag_list)
    results_list = list(result)
    print("Results: ", results_list)
    print(len(results_list))
    # tags = results_list['tags']
    if result and collection_name == 'apis':
        return render_template('results_a.html', results=results_list, size=len(results_list))
    elif result and collection_name == 'mashups':
        return render_template('results_m.html', results=results_list, size=len(results_list))
    else:
        return render_template('error.html', error=True)


@app.errorhandler(404)
def page_not_found(e):
    """
    Render the error page
    :param e:
    :return:
    """
    return render_template('error.html', error=True)


if __name__ == '__main__':
    app.run(debug=True)
