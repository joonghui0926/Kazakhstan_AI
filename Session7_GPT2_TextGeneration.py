# Session 7: GPT-2 Text Generation
from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')
prompt = "The future of AI is"
outputs = generator(prompt, max_length=50, num_return_sequences=1)
print(outputs[0]['generated_text'])