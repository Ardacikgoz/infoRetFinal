from transformers import AutoTokenizer, AutoModel

AutoTokenizer.from_pretrained("distilbert-base-uncased")
AutoModel.from_pretrained("distilbert-base-uncased")