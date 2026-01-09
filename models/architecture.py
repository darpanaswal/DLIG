from models.model_manager import ModelManager

model_manager = ModelManager()
model, tokenizer = model_manager.load_model_and_tokenizer()

print(model)