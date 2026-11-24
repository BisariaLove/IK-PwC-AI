import boto3
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")

# Load image from file
with open("IMG_0770.jpeg", "rb") as f:
    image_bytes = f.read()


try:
    response = client.converse(
        modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{
            "role": "user",
            "content": [
                {"image": {
                    "format": "jpeg",
                    "source": {"bytes": image_bytes}
                }
                },
                {"text": "What is in this image?"}
            ]
        }]
    )
    print(response["output"]["message"]["content"][0]["text"])
except ClientError as e:
    print("Request ID:", e.response["ResponseMetadata"].get("RequestId"))
    print("AWS error:", e.response["Error"])
    raise
