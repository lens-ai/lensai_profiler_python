import os
import tarfile
import gzip
import shutil
import requests
import time

def tar_and_gzip_folder(folder_path, output_filename):
    """
    Tar and gzip a folder.
    Args:
        folder_path (str): The path to the folder to compress.
        output_filename (str): The name of the output tar.gz file.
    Returns:
        str: The path to the tar.gz file.
    """

    # Ensure folder_path is a string (path to the folder)
    if not isinstance(folder_path, str):
        raise TypeError(f"Expected folder_path to be a string, got {type(folder_path)} instead.")
 
    # Ensure the folder path exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder path '{folder_path}' does not exist or is not a directory.")

    # Create paths for the tar and gzip files in the /tmp/ directory
    tar_path = f'/tmp/{output_filename}.tar'
    gzip_path = f'/tmp/{output_filename}.tar.gz'

    # Create a tar file
    with tarfile.open(tar_path, 'w') as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    
    # Gzip the tar file
    with open(tar_path, 'rb') as f_in:
        with gzip.open(gzip_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Remove the intermediate tar file
    os.remove(tar_path)
    return gzip_path


def post_file_to_endpoint(file_path, endpoint_url, sensor_id, timestamp, metrictype):
    """
    Post a file to a given endpoint with headers.

    Args:
        file_path (str): The path to the file to post.
        endpoint_url (str): The URL of the endpoint.
        sensor_id (str): The value for the 'sensorid' header.
        timestamp (int): The Unix timestamp for the 'timestamp' header.
    """
    headers = {
        'metrictype': metrictype,    
        'sensorid': sensor_id,
        'timestamp': str(timestamp)  # Unix timestamp as string
    }

    with open(file_path, 'rb') as file:
        response = requests.post(endpoint_url, headers=headers, files={'file': file})

    if response.status_code == 200:
        print(f"File {file_path} successfully posted to {endpoint_url}.")
    else:
        print(f"Failed to post file. Status code: {response.status_code}, Response: {response.text}")
