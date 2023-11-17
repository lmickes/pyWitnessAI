import os.path
import pandas as pd
import ast


def flatten_keys(structure):
    keys = flatten_recursive(structure)
    return [k[0:-4] for k in keys]


def flatten_data(structure):
    data = []

    # If the structure is a list, extend the flattened result of each element
    if isinstance(structure, list):
        for element in structure:
            data.extend(flatten_data(element))

    # If the structure is a dictionary, extend the flattened result of each value
    elif isinstance(structure, dict):
        for key in structure:
            data.extend(flatten_data(structure[key]))

    # If the structure is a simple value, wrap it in a list (Make it always return a list)
    else:
        data = [structure]

    return data


def flatten_recursive(structure):
    keys = []

    # If the structure is a list, append the index and recursively obtain further keys
    if isinstance(structure, list):
        for i, e in zip(range(0, len(structure)), structure):
            for dk in flatten_recursive(e):
                keys.append(str(i) + "_" + dk)

    # If the structure is a dictionary, append the key and recursively obtain further keys
    elif isinstance(structure, dict):
        for k in structure:
            for dk in flatten_recursive(structure[k]):
                keys.append(k + "_" + dk)

    # If the structure is a simple value, return a single key indicating a value
    else:
        keys = ['val']

    return keys


class DataFlattener:
    def __init__(self, dataframe, column_to_be_processed, preprocess_lists=False):
        self.df = dataframe
        if preprocess_lists:
            self.preprocess_list_columns(column_to_be_processed)

    @staticmethod
    def convert_string_to_list(row):
        # Convert a string data into a list
        try:
            return ast.literal_eval(row)
        except (ValueError, SyntaxError):
            return None

    def preprocess_list_columns(self, columns):
        for col in columns:
            self.df[col] = self.df[col].apply(self.convert_string_to_list)

    def flatten_column(self, column_name):
        max_items = self.df[column_name].apply(lambda x: len(x) if x is not None else 0).max()
        for i in range(max_items):
            new_col_name = f'{column_name}_{i+1}'
            self.df[new_col_name] = self.df[column_name].apply(
                lambda x: x[i] if x is not None and i < len(x) else None
            )
        self.df.drop(column_name, axis=1, inplace=True)

    def flatten_complex_column(self, column_name):
        flattened_data = self.df[column_name].apply(flatten_data)
        sample_data = self.df[column_name].iloc[0]
        if sample_data is not None:
            flattened_keys = flatten_keys(flatten_recursive(sample_data))
            flattened_df = pd.DataFrame(flattened_data.tolist(), columns=flattened_keys)
            self.df = pd.concat([self.df.drop(column_name, axis=1), flattened_df], axis=1)

    def flatten(self, columns):
        # Flatten each complex column individually
        for column_name in columns:
            self.flatten_complex_column(column_name)
        return self.df

# converters = {
#     'mtcnn_confidence': convert_string_to_list,
#     'lineup_similarities': convert_string_to_list,
# }


# Assuming that your data has complex nested structures in the 'complex_column'
# You can preprocess those columns and then flatten them
to_be_processed = ['mtcnn_confidence', 'lineup_similarities']
df = pd.read_csv('./results/analyzed_data.csv')
flattener = DataFlattener(df, to_be_processed, preprocess_lists=True)
flat_df = flattener.flatten(to_be_processed)

# Save the flattened DataFrame back to a CSV
flat_df.to_csv(os.path.join('results', 'flattened_data_1.csv'), index=False)


