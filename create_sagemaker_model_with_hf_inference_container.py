import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import logging
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s: %(message)s')

# Retrieve IAM role and model data path from environment variables
sagemaker_role = os.getenv('SAGEMAKER_IAM_ROLE')
model_data_s3_path = os.getenv('MODEL_DATA_S3_PATH')

logging.info("Starting SageMaker model creation")

try:
    # Initialize the Hugging Face SageMaker Model
    huggingface_model = HuggingFaceModel(
        model_data=model_data_s3_path,  # S3 path to your model artifacts
        role=sagemaker_role,  # IAM role with permissions to create an endpoint
        transformers_version='4.28.1',  # Transformers version used
        pytorch_version='2.0.0',  # PyTorch version used
        py_version='py310',  # Python version
        entry_point='inference.py'  # Local path to the inference script for SageMaker's Hugging Face container
    )
    logging.info("SageMaker model created successfully")

    try:
        # Deploy the model to create an endpoint
        logging.info("Starting SageMaker endpoint deployment")
        predictor = huggingface_model.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.large'
        )
        logging.info(f"SageMaker endpoint deployed successfully, endpoint name: {predictor.endpoint_name}")
    except Exception as e:
        logging.error(f"Error deploying SageMaker endpoint: {e}")

except Exception as e:
    logging.error(f"Error creating SageMaker model: {e}")
