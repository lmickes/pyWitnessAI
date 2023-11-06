import os.path
import pandas as pd
import ast


class DataFlattener:
    def __init__(self, dataframe, column_to_be_processed, preprocess_lists=False):
        self.df = dataframe
        if preprocess_lists:
            self.preprocess_list_columns(column_to_be_processed)

    def preprocess_list_columns(self, column_to_be_processed):
        # Add all columns here that need conversion from string to list
        list_columns = column_to_be_processed
        for col in list_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self.safe_literal_eval)

    @staticmethod
    def safe_literal_eval(row):
        try:
            return ast.literal_eval(row)
        except (ValueError, SyntaxError):
            return []

    def flatten_column(self, column_name, new_column_template):
        #  Extract the maximum number of items in the nested lists of the specified column
        max_items = self.df[column_name].apply(lambda x: len(x)).max()

        #  Create new columns for each item
        for i in range(max_items):
            #  New column names
            new_col_name = f'{new_column_template}_{i+1}'
            #  Assign the values from the nested list
            self.df[new_col_name] = self.df[column_name].apply(lambda x: x[i] if i < len(x) else None)

        #  Drop the original column
        self.df.drop(column_name, axis=1, inplace=True)

    def flatten_multidimensional_column(self, column_name, new_column_template):
        #  Get the max dimension
        dimensions = self.df[column_name].apply(lambda x: (len(x), len(x[0]) if x else 0)).max()
        print(f'dimension of this column is {dimensions}')

        #  Create new columns for each item in the 2D list
        for i in range(dimensions[0]):
            for j in range(dimensions[1]):
                new_col_name = f'{new_column_template}_face{i+1}_face{j+1}'
                self.df[new_col_name] = self.df[column_name].apply(
                    lambda x: x[i][j] if i < len(x) and j < len(x[i]) else None)

        self.df.drop(column_name, axis=1, inplace=True)

    def flatten(self):
        #  We should know which column needs to be flattened
        self.flatten_column('mtcnn_confidence', 'mtcnn_confidence_value')
        self.flatten_multidimensional_column('lineup_similarities',
                                             'lineup_similarities_between')
        return self.df


# How to use the Flattener
df = pd.read_csv('./results/analyzed_data.csv')

to_be_processed = ['mtcnn_confidence', 'lineup_similarities']
column_to_be_flattened = ['mtcnn_confidence', 'lineup_similarities']

flattener = DataFlattener(df, to_be_processed, preprocess_lists=True)
flat_df = flattener.flatten()

# Save the flattened DataFrame back to a CSV
flat_df.to_csv(os.path.join('results', 'flattened_data.csv'), index=False)
