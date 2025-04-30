import os
import csv

def add_suffix_to_csv(input_csv_path, new_size):
    """
    Reads a CSV file, adds a suffix to the first column of each row (excluding the header), 
    and writes the modified data to a new CSV file in a folder named after the input folder 
    with the new size appended.

    Args:
        input_csv_path (str): Path to the input CSV file.
        new_size (tuple): A tuple containing the new size (width, height) to append as a suffix.
    """
    suffix = f"_{new_size[0]}x{new_size[1]}"
    
    # Determine the output folder and file path
    input_folder = os.path.dirname(input_csv_path)
    folder_name = os.path.basename(input_folder)
    output_folder = os.path.join(os.path.dirname(input_folder), f"{folder_name}_{new_size[0]}x{new_size[1]}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_csv_name = os.path.basename(input_csv_path)
    input_csv_base, input_csv_ext = os.path.splitext(input_csv_name)
    output_csv_name = f"{input_csv_base}{suffix}{input_csv_ext}"
    output_csv_path = os.path.join(output_folder, output_csv_name)
    
    with open(input_csv_path, mode='r', newline='', encoding='utf-8') as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        rows = list(reader)
        
    # Add suffix to the first column of each row, excluding the header
    modified_rows = []
    for i, row in enumerate(rows):
        if row:  # Ensure the row is not empty
            if i == 0:  # Skip the header row
                modified_rows.append(row)
            else:
                row[0] = f"{row[0]}{suffix}"
                modified_rows.append(row)
    
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerows(modified_rows)

# Example usage
input_csv = '/dtu/blackhole/07/203495/ADLCV-project/data/ISIC_2019_Training_Input/generated_descriptions_isic_5000.csv'
new_size = (128, 128)

add_suffix_to_csv(input_csv, new_size)