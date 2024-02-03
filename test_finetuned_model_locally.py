import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('./fine_tuned_model/', local_files_only=True)
tokenizer = DistilBertTokenizer.from_pretrained('./fine_tuned_model/', local_files_only=True)

# Your input text
text = "This is a great movie!"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits from the model output
logits = outputs.logits

# Convert logits to probabilities (optional)
probabilities = torch.softmax(logits, dim=1)

# Print the logits or probabilities
print("Logits:", logits)
print("Probabilities:", probabilities)

# Determine the predicted sentiment (assuming index 0 is negative and index 1 is positive sentiment)
predicted_sentiment = "positive" if logits[0][1] > logits[0][0] else "negative"
print("Predicted sentiment:", predicted_sentiment)
