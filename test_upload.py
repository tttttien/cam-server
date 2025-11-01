import boto3
try:
    s3 = boto3.client(
        's3',
        endpoint_url='https://bwmqzqgnouisgshuprhh.storage.supabase.co/storage/v1/s3',
        aws_access_key_id='0ce4e6b6d05b9bf274d7a554d1cee534',
        aws_secret_access_key='c596dda78c2c7dfdd351680b19b30b72fa8965700999460686abc7d7e66894d2',
    )
    with open('/tmp/test.jpg','wb') as f: f.write(b'test')  # tạo file test nhỏ
    with open('/tmp/test.jpg','rb') as f:
        s3.put_object(Bucket='fire', Key='test/test.jpg', Body=f, ContentType='image/jpeg')
    print('S3 put_object ok')
except Exception as e:
    print('S3 test error:', e)