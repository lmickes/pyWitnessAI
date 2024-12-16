import os
import shutil
import csv


def organize_photos(photos_directory, results_file, column_name):
    # Read the CSV file
    with open(results_file, 'r') as file:
        csv_reader = csv.DictReader(file)

        # Create a dictionary to store the mapping of photo names to their corresponding results
        photo_results = {}
        for row in csv_reader:
            photo_number = row['number'].split('(')[1].split(')')[0]  # Extract the number from the photo name
            photo_name = f"image ({photo_number}).jpg"  # Construct the actual photo name
            result = row[column_name]
            photo_results[photo_name] = result

    # Create directories based on the unique values in the specified column
    unique_results = set(photo_results.values())
    for result in unique_results:
        directory_name = f"{column_name}_{result}"
        os.makedirs(directory_name, exist_ok=True)

    # Assign each photo to the right directory based on its result
    for photo_name, result in photo_results.items():
        source_path = os.path.join(photos_directory, photo_name)
        destination_path = os.path.join(f"{column_name}_{result}", photo_name)
        shutil.copy(source_path, destination_path)

    print("Photo organization completed.")



# Example usage
photos_directory = "D:/MscPsy/Data/Colloff2021/FillerLibrary/"
csv_file = "D:/MscPsy/Data/Colloff2021/results/R_Analysis/merged_data.csv"
column_name = "similarity_group_to_perp_frs"

organize_photos(photos_directory, csv_file, column_name)