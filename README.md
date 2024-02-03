# DistilBERT Sentiment Analysis on IMDB Dataset

This project involves fine-tuning the DistilBERT model for sentiment analysis on the IMDB dataset and deploying it using AWS SageMaker. The model is fine-tuned to predict the sentiment of a given text as positive or negative.

## Project Overview

- **Model Training**: The DistilBERT model was fine-tuned using the IMDB dataset on Google Colab, utilizing the available GPU resources for efficient training.

- **Local Testing**: The trained model was tested locally to ensure accuracy and performance.

- **AWS Deployment**: The model was deployed to AWS using SageMaker, which simplifies the deployment process and provides a managed environment.

- **Endpoint Testing**: The deployed model was tested via the SageMaker endpoint to validate its functionality in the cloud.

## Repository Structure

- `fine_tuned_model/`: Contains the fine-tuned DistilBERT model and tokenizer files (added to `.gitignore`, but you can obtain them by running the `DistilBERT_finetune_imdb.ipynb` notebook)
- `DistilBERT_finetune_imdb.ipynb`: Colab notebook for fine-tuning the model
- `create_sagemaker_model_with_hf_inference_container.py`: Script to create and deploy the model to AWS SageMaker.
- `test_finetuned_model_locally.py`: Script to test the model locally.
- `inference_requests.py`: Script to make inference requests to the SageMaker endpoint.
- `requirements.txt`: Lists the necessary Python packages for the project.

## Setup and Installation

1. **Clone the repository**:

    git clone https://github.com/ML-D00M/FineTuned-DistilBERT-IMDB.git

2. **Set up a virtual environment** (optional but recommended):

    python -m venv venv
source venv/bin/activate # On Windows use venv\Scripts\activate

3. **Install the requirements**:

    pip install -r requirements.txt

4. **Prepare the model directory**:

    Since the `fine_tuned_model/` directory is not included in the repository, you will need to obtain the fine-tuned model and tokenizer by running the `DistilBERT_finetune_imdb.ipynb` notebook (fine-tuning took about 1-2 hours in Colab with T4 GPU runtime)


## Running the Project
- To test the model locally:

    python test_finetuned_model_locally.py

- To deploy the model to AWS SageMaker:

    python create_sagemaker_model_with_hf_inference_container.py

- To make an inference request to the SageMaker endpoint:

    python inference_requests.py

## Contact
If you have any questions or comments about the project, please feel free to reach out.