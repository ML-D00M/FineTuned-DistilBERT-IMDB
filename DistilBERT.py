from transformers import DistilBertModel, DistilBertTokenizer
import torch

# Load the model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def predict(input_text):
    # Preprocess the input
    inputs = tokenizer(input_text, return_tensors="pt")

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process and return the output
    # (custom logic depending on your application)
    return outputs

# Example usage of the predict function
if __name__ == "__main__":
    sample_text = "Hello, world!"
    result = predict(sample_text)
    print(result)
