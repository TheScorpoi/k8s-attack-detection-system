from google.cloud import storage
from google.oauth2 import service_account
import json


filename = './rosy-fiber-410201-b6502b7424f9.json'

# Load the service account key JSON file contents
with open(filename, 'r') as file:
    gcs_credentials = json.load(file)

# Use the credentials to configure service account credentials
credentials = service_account.Credentials.from_service_account_info(gcs_credentials)
storage_client = storage.Client(credentials=credentials, project=credentials.project_id)

 
bucket_name = 'models-aveiro'

destination_blob_name = 'test.png'

source_file_name = 'test.png'

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

print("Uploading file to bucket...")
#upload_blob(bucket_name, source_file_name, destination_blob_name)
