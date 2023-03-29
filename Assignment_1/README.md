
# Brewery Application PA1

This application allows you to search for breweries near a given location and view their information, including ratings. People who are travelling to a new place can use this to find information about the breweries present near that location and plan their iteneraries.


## API Reference

#### homepage

```http
  GET /
```


#### Get Breweries
Takes the location and gives a list of brewries around that location along with their addresses and websites.

```http
  POST /breweries/${latlng}
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `latlng`      | `string` | **Required**. location |

#### Get Ratings

Takes a brewery and returns its google and yelp ratings.

```http
  POST /reviews/${conversation_id}/${brewery_id}
```
| Parameter | Type     | Description                              |
| :-------- | :------- |:-----------------------------------------|
| `brewery_id`      | `string` | **Required**. id of the selected brewery |
| `conversation_id`      | `string` | **Required**. id of the conversation     |


## Authors

- [@grad-kushal](https://github.com/grad-kushal)


## Environment Variables

To run this project, you will need to set the following environment variables in your terminal

`export FLASK_APP=app.py`

`export FLASK_ENV=development`


## Run Locally

Clone the project

```bash
  git clone https://github.com/grad-kushal/CSCI724/tree/master
```

Go to the project directory

```bash
  cd Assignment_1
```

Install dependencies

```bash
  pip install <dependency-name>
```

Start the server

```bash
  flask run
```

Go to http://127.0.0.1:5000/ in your browser

