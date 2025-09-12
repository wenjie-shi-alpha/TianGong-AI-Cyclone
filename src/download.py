import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))

bucket = "noaa-oar-mlwp-data"

# 列出部分对象
resp = s3.list_objects_v2(Bucket=bucket, MaxKeys=10)
for obj in resp.get("Contents", []):
    print(obj["Key"])
