import torch
from transformers import DistilBertModel

def save_model(output_path):
    # Load the pre-trained DistilBERT model
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    # Save the model to the specified path
    torch.save(model.state_dict(), output_path)

if __name__ == "__main__":
    # Define the path where you want to save the model
    output_path = 'model.pth'
    save_model(output_path)
