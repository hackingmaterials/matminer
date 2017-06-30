def flattenDict(d, result=None):
    """
    Function to flatten a JSON dictionary. Taken from: https://gist.github.com/higarmi/6708779

    example: The following JSON document:
    {"maps":[{"id1":"blabla","iscategorical1":"0", "perro":[{"dog1": "1", "dog2": "2"}]},{"id2":"blabla","iscategorical2":"0"}],
    "masks":{"id":"valore"},
    "om_points":"value",
    "parameters":{"id":"valore"}}

    will have the following output:
    {'masks.id': 'valore', 'maps.iscategorical2': '0', 'om_points': 'value', 'maps.iscategorical1': '0',
    'maps.id1': 'blabla', 'parameters.id': 'valore', 'maps.perro.dog2': '2', 'maps.perro.dog1': '1', 'maps.id2': 'blabla'}
    """
    if result is None:
        result = {}
    for key in d:
        value = d[key]
        if isinstance(value, dict):
            value1 = {}
            for keyIn in value:
                value1[".".join([key,keyIn])]=value[keyIn]
            flattenDict(value1, result)
        elif isinstance(value, (list, tuple)):
            for indexB, element in enumerate(value):
                if isinstance(element, dict):
                    value1 = {}
                    index = 0
                    for keyIn in element:
                        newkey = ".".join([key,keyIn])
                        value1[".".join([key,keyIn])]=value[indexB][keyIn]
                        index += 1
                    for keyA in value1:
                        flattenDict(value1, result)
        else:
            result[key]=value
    return result