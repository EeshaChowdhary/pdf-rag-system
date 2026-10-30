import os
import boto3


def s3_enabled():
    return bool(os.getenv("S3_BUCKET"))


def save_pdf(name, data):
    """Save an uploaded PDF to S3. Does nothing if S3_BUCKET is not set."""
    if not s3_enabled():
        return False
    client = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
    client.put_object(
        Bucket=os.environ["S3_BUCKET"],
        Key=f"uploads/{name}",
        Body=data,
        ContentType="application/pdf",
    )
    return True