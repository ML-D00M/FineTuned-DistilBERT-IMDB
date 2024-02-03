import boto3
import json
import os
from dotenv import load_dotenv


# Create a SageMaker runtime client with the AWS region
client = boto3.client('sagemaker-runtime', region_name='eu-north-1')

# Specify the name of your deployed endpoint from the .env variable
endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')

# Prepare your input text that you want to analyze for sentiment by DistilBERT fine-tuned on IMDB dataset
input_text = "I loved this movie, it was fantastic!"

# Format the input data as a JSON string. Adjust this format if the model expects something different.
input_data_json = json.dumps({"inputs": input_text})

# Invoke the SageMaker endpoint for inference
response = client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',  # The content type may vary based on the model's requirements
    Body=input_data_json
)

# Decode and parse the inference result from the response
result = json.loads(response['Body'].read().decode('utf-8'))

# Assuming the model returns a human-readable sentiment analysis result in the response
print(f"Response from model: {result}")
