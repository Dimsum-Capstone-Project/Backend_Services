from google.cloud import storage
import os
from dotenv import load_dotenv

load_dotenv()

def upload_to_gcs(file_name, content, bucket_name=None):
    # Inisialisasi client GCS
    client = storage.Client()
    if bucket_name is None:
        bucket = client.bucket(os.getenv("GCS_BUCKET_NAME", "dimsum_palm_private"))
    else:
        bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Upload data ke GCS
    blob.upload_from_string(content)
    if bucket_name is None:
        print(f"File {file_name} berhasil di-upload ke bucket {os.getenv('GCS_BUCKET_NAME', 'dimsum_palm_private')}")
    else:
        print(f"File {file_name} berhasil di-upload ke bucket {bucket_name}")
    
    
def get_from_gcs(file_name, bucket_name=None):
    # Inisialisasi client GCS
    client = storage.Client()
    if bucket_name is None:
        bucket = client.bucket(os.getenv("GCS_BUCKET_NAME", "dimsum_palm_private"))
    else:
        bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    if bucket_name is None:
        print(f"File {file_name} berhasil di-download dari bucket {os.getenv('GCS_BUCKET_NAME', 'dimsum_palm_private')}")
    else:
        print(f"File {file_name} berhasil di-download dari bucket {bucket_name}")
    
    # Download data dari GCS

    try:
        content = blob.download_as_bytes()
        if content is None:
            print(f"File {file_name} tidak ditemukan di bucket {os.getenv('GCS_BUCKET_NAME', 'dimsum_palm_private')}")
            return None
    except Exception as e:
        print(f"Terjadi kesalahan saat mendownload file {file_name} dari bucket {os.getenv('GCS_BUCKET_NAME', 'dimsum_palm_private')}: {e}")
        return None

    return content

def delete_all_files_in_gcs(bucket_name=None):
    # Inisialisasi client GCS
    client = storage.Client()
    if bucket_name is None:
        bucket = client.bucket(os.getenv("GCS_BUCKET_NAME", "dimsum_palm_private"))
    else:
        bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()

    for blob in blobs:
        blob.delete()
        if bucket_name is None:
            print(f"File {blob.name} berhasil dihapus dari bucket {os.getenv('GCS_BUCKET_NAME', 'dimsum_palm_private')}")
        else:
            print(f"File {blob.name} berhasil dihapus dari bucket {bucket_name}")
            

# bucket_name = "dimsum_palm_private"
# file_name = "test.txt"
# content = "Data untuk disimpan di GCS"

# upload_to_gcs(bucket_name, file_name, content)
# content = get_from_gcs(bucket_name, file_name)
# print(content)
