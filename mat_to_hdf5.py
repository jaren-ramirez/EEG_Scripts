"""
Converts eeg data from matlab to hdf5 for ease of use.
"""


import argparse
import os
import re
from scipy.io import loadmat
import numpy as np
import h5py

def _convert_strings_to_utf8(arr):
    """
    Convert an array of arrays containing strings to a single array of UTF-8 encoded strings.
    """
    flat_list = [item[0] for item in arr]  # Flatten the array of arrays
    str_array = np.array(flat_list, dtype='S')  # Convert to UTF-8 encoded bytes
    return str_array

def convert_mat_to_hdf5(output_name, mat):
    """
    Converts a MATLAB (.mat) file to HDF5 (.hdf5) format.
    """
    with h5py.File(output_name, "w") as f:
        for key, value in mat.items():
            if key in ["__header__", "__version__", "__globals__"]:
                continue  # Skip metadata keys
            
            key = re.sub(r'\d+$', '', key)  # Remove trailing digits from key names
            
            if isinstance(value, np.ndarray) and value.dtype.type is np.object_:
                try:
                    utf8_strings = _convert_strings_to_utf8(value)
                    f.create_dataset(key, data=utf8_strings)
                except Exception as e:
                    print(f"Could not convert {key} to UTF-8 strings:", e)
            elif isinstance(value, np.ndarray) or isinstance(value, str):
                if isinstance(value, str):
                    value = np.string_(value)  # Convert string to NumPy string format
                f.create_dataset(key, data=value)
                
def get_mat_files(source: str) -> list:
    """
    Retrieves a list of .mat files from the specified source.
    """
    if not os.path.exists(source):
        raise ValueError("File or path does not exist")
    
    files = []
    if os.path.isfile(source) and source.endswith('.mat'):
        files = [{"name": os.path.splitext(os.path.basename(source))[0], "mat": loadmat(source)}]
    elif os.path.isdir(source):
        for file_name in os.listdir(source):
            if file_name.endswith('Imputed.mat'):
                file_path = os.path.join(source, file_name)
                name = re.match(r'(song[0-9][0-9]_Imputed)(\.mat$)', file_name)
                if name:
                    mat_content = loadmat(file_path)
                    files.append({"name": name.group(1), "mat": mat_content})
    if not files:
        raise ValueError("No valid .mat files found in the specified source.")
    return files

def mat_to_hdf5(mat_files: list, destination: str):
    """
    Converts .mat files to .hdf5 format and saves them to the specified destination.
    """
    if not os.path.exists(destination):
        os.makedirs(destination)
    for mat_file in mat_files:
        hdf5_name = f"{mat_file['name']}.hdf5"
        hdf5_path = os.path.join(destination, hdf5_name)
        convert_mat_to_hdf5(hdf5_path, mat_file['mat'])


def parse_arguments():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Converts .mat data files to .hdf5 format.")
    parser.add_argument("-src", "--source", type=str, default=os.getcwd(),
                        help="The source directory or file. Defaults to the current directory.")
    parser.add_argument("-dst", "--destination", type=str, default=os.path.join(os.getcwd(), "hdf5"),
                        help="The destination directory for .hdf5 files. Defaults to a 'hdf5' directory in the current working directory.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    try:
        mat_files = get_mat_files(args.source)
        mat_to_hdf5(mat_files, args.destination)
        print("Conversion completed successfully.")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()