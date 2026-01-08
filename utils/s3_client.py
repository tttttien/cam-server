import os
import traceback
from typing import Optional, Dict, Any
from config import S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except ImportError:
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

_s3_client = None

def get_s3_client():
    """
    Initializes and returns a singleton instance of the Boto3 S3 client.
    
    Returns:
        botocore.client.S3: The S3 client instance, or None if boto3 is not installed.
    """
    global _s3_client
    if _s3_client:
        return _s3_client
    if boto3 is None:
        print("❌ boto3 is not installed. S3 functionality disabled.")
        return None
    
    _s3_client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    return _s3_client

def upload_file_to_s3(path: str, object_name: str, content_type: str, metadata: Optional[Dict[str, str]] = None) -> bool:
    """
    Uploads a local file to the configured S3 bucket.

    Args:
        path (str): The local file path to upload.
        object_name (str): The destination object name (key) in the bucket.
        content_type (str): The MIME type of the file.
        metadata (Dict[str, str], optional): Custom metadata to attach to the S3 object.

    Returns:
        bool: True if the upload was successful, False otherwise.
    """
    client = get_s3_client()
    if client is None:
        return False
    
    try:
        # Note: boto3's upload_file is synchronous
        client.upload_file(
            path,
            S3_BUCKET_NAME,
            object_name,
            ExtraArgs={
                "ContentType": content_type,
                "Metadata": metadata or {}
            },
        )
        return True
    except Exception as e:
        print(f"❌ S3 upload error Details: {type(e).__name__} - {str(e)}")
        traceback.print_exc()
        return False
