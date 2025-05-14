# Session 5: NLP Basics - Tokenization & Embedding
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

texts = ["Hello, AI world!", "This is a test."]
inputs = tokenizer(texts, padding=True, return_tensors="tf")
outputs = model(**inputs)
print("Embedding shape:", outputs.last_hidden_state.shape)