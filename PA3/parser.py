import json


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


def read_mashup_data(filename):
    """
    Read the mashup data from the file
    :param filename: The name of the file to read
    :return: The JSON object containing the mashup data
    """
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()

    mashup_records = []

    for line in lines:
        fields = line.strip().split('$#$')
        mashup_record = {
            'id': fields[0],
            'title': fields[1],
            'summary': fields[2],
            'rating': float(fields[3]) if fields[3] else 0,
            'name': fields[4],
            'label': fields[5],
            'author': fields[6],
            'description': fields[7],
            'type': fields[8],
            'downloads': fields[9],
            'useCount': fields[10],
            'sampleUrl': fields[11],
            'dateModified': fields[12],
            'numComments': fields[13],
            'commentsUrl': fields[14],
            'tags': fields[15].split('###'),
            'apis': [],
            'updated': fields[17]
        }
        for ap in fields[16].split('###'):
            ap_fields = ap.split('$$$')
            if len(ap_fields) == 2:
                mashup_record['apis'].append({
                    'name': ap_fields[0],
                    'url': ap_fields[1]
                })
        mashup_records.append(mashup_record)
    mashup_record_json = json.dumps(mashup_records)
    return mashup_records


def read_api_data(filename):
    """
    Read the API data from the file
    :param filename: The name of the file to read
    :return: The JSON object containing the API data
    """
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        lines = f.readlines()

    api_records = []

    for line in lines:
        fields = line.strip().split('$#$')
        api_record = {
            'id': fields[0],
            'title': fields[1],
            'summary': fields[2],
            'rating': float(fields[3]) if fields[3] else 0,
            'name': fields[4],
            'label': fields[5],
            'author': fields[6],
            'description': fields[7],
            'type': fields[8],
            'downloads': fields[9],
            'useCount': fields[10],
            'sampleUrl': fields[11],
            'downloadUrl': fields[12],
            'dateModified': fields[13],
            'remoteFeed': fields[14],
            'numComments': fields[15],
            'commentsUrl': fields[16],
            'tags': [tag.lower() for tag in fields[17].split('###')],
            'category': fields[18].lower(),
            'protocols': fields[19],
            'serviceEndpoint': fields[20],
            'version': fields[21],
            'wsdl': fields[22],
            'dataFormats': fields[23],
            'apiGroups': fields[24],
            'example': fields[25],
            'clientInstall': fields[26],
            'authentication': fields[27],
            'ssl': fields[28],
            'readonly': fields[29],
            'VendorApiKits': fields[30],
            'CommunityApiKits': fields[31],
            'blog': fields[32],
            'forum': fields[33],
            'support': fields[34],
            'accountReq': fields[35],
            'commercial': fields[36],
            'provider': fields[37],
            'managedBy': fields[38],
            'nonCommercial': fields[39],
            'dataLicensing': fields[40],
            'fees': fields[41],
            'limits': fields[42],
            'terms': fields[43],
            'company': fields[44],
            'updated': fields[45]
        }

        api_records.append(api_record)
    api_records_json = json.dumps(api_records)
    return api_records


def create_matrix(mashup_records, api_records):
    matrix = [[0 for x in range(len(api_records))] for y in range(len(mashup_records))]
    for i in range(len(mashup_records)):
        mashup = mashup_records[i]
        for j in range(len(api_records)):
            api = api_records[j]
            mashup_apis = mashup['apis']
            if mashup_apis:
                mashup_api_names = [a.split("$$$")[0] for a in mashup['apis']]
                if api['name'] in mashup_api_names:
                    matrix[i][j] = 1
                    # print("Mashup: " + mashup['title'] + " API: " + api['title'])
                    # print(i, j)

    return matrix


def main():
    """
    Main function
    :return: None
    """
    # Read the API data from the file
    api_records = read_api_data('data/api.txt')
    api_records_json = json.dumps(api_records)

    mashup_records = read_mashup_data('data/mashup.txt')
    mashup_records_json = json.dumps(mashup_records)

    mat = create_matrix(mashup_records, api_records)

    print(mat[7391][3888])


if __name__ == '__main__':
    main()
