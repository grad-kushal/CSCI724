
# API/Mashup Search Engine

This is a web application that allows users to search for APIs based on their titles, categories, tags, ratings, and descriptions. The application is built using Python Flask framework and MongoDB for data storage.


## API Reference

### Landing Page
Provides two options to query the system: 
```http
  GET /
```

    1. Keywords search
    2. Advanced search



### Advanced Search
Shows the advanced search options to the user

```http
  GET /advanced_search
```

### Keywords Search
Shows the keywords search options to the user

```http
  GET /keywords_search
```


### Get Advanced Search Results

Takes the user provided advanced criteria and returns the results matching it.

```http
  GET /results
```


### Get Keywords Search Results

Takes the user provided advanced criteria and returns the results matching it.

```http
  GET /search_results
```


## Environment Variables

To run this project, you will need to set the following environment variables in your terminal

` export FLASK_APP=app.py`



## Run Locally

Clone the project

```bash
  git clone https://github.com/grad-kushal/CSCI724.git
```

Go to the project directory

```bash
  cd PA3
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  flask run
```

Go to http://127.0.0.1:5000/ in your browser

