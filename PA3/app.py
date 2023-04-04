import parser

from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap

import database

app = Flask(__name__)
bootstrap = Bootstrap(app)

mydb = database.connect()
if len(mydb.list_collection_names()) != 2:
    database.delete_collection(mydb, 'mashups')
    database.delete_collection(mydb, 'apis')
    print("Obsolete database. Fixing it...")
    mashup_data, api_data = parser.read_data()
    database.insert_documents(mydb, 'mashups', mashup_data)
    database.insert_documents(mydb, 'apis', api_data)


@app.route('/')
def index():
    """
    Render the index page
    :return:
    """
    return render_template('landing.html')


@app.route('/advanced_search', methods=['GET'])
def search_a():
    """
    Render the index page
    :return:
    """
    return render_template('index.html')


@app.route('/keyword_search', methods=['GET'])
def search():
    """
    Render the search page
    :return:
    """
    return render_template('search.html')


@app.route('/search_results', methods=['GET', 'POST'])
def search_results():
    """
    Render the keywords_search results page
    :return:
    """
    print(request.form)
    collection_name = request.form['search-type']
    keywords = request.form['keywords']
    print(keywords)
    keywords = keywords.split()
    result = database.get_documents_by_keywords(mydb, collection_name, keywords)
    results_list = list(result)
    print(len(results_list), " results found.")
    if result and collection_name == 'apis':
        return render_template('results_a.html', results=results_list, size=len(results_list))
    elif result and collection_name == 'mashups':
        return render_template('results_m.html', results=results_list, size=len(results_list))
    else:
        return render_template('error.html', error=True)


@app.route('/results', methods=['POST'])
def results():
    """
    Render the results page
    :return:
    """
    print("Advanced search")
    collection_name = request.form['search-type']
    updated_year = request.form['updated-year']
    category = request.form['category']
    category = category.strip()
    rating = request.form['rating']
    rating = rating.strip() if rating else 0.0
    rating = float(rating)
    rating_comparison = request.form['rating-comparison']
    tags = request.form['tags']
    protocols = request.form['protocols']
    tag_list = []
    if tags:
        tag_list = tags.split(',')
    result = database.get_documents(mydb, collection_name, updated_year, category, float(rating), rating_comparison,
                                    tag_list, protocols)
    results_list = list(result)
    print(len(results_list), " results found.")
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


@app.errorhandler(403)
def page_not_found(e):
    """
    Render the error page
    :param e:
    :return:
    """
    return render_template('error.html', error=True)


@app.errorhandler(400)
def page_not_found(e):
    """
    Render the error page
    :param e:
    :return:
    """
    return render_template('error.html', error=True)


if __name__ == '__main__':
    app.run(debug=True)
