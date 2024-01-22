def flatten_keys(structure):
    keys = flatten_recursive(structure)
    return [k[0:-4] for k in keys]


def flatten_data(structure):

    data = []

    if type(structure) == list:
        for i, e in zip(range(0, len(structure)), structure):
            for d in flatten_data(e):
                data.append(d)
    elif type(structure) == dict:
        for k in structure:
            for d in flatten_data(structure[k]):
                data.append(d)
    else:
        data = [structure]

    return data


def flatten_recursive(structure):

    keys = []

    if type(structure) == list:
        for i, e in zip(range(0, len(structure)), structure):
            for dk in flatten_recursive(e):
                keys.append(str(i)+"_"+dk)
    elif type(structure) == dict:
        for k in structure:
            for dk in flatten_recursive(structure[k]):
                keys.append(k+"_"+dk)
    else:
        keys = ['val']

    return keys
