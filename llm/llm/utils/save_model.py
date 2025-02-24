import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

def save_quantized_model(model_name = "gpt2", quantized_model_path = "gpt2_quantized.pth"):
    # Load GPT-2 model and tokenizer
    torch.backends.quantized.engine = 'qnnpack'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)  # For MRPC classification
    # Ensure padding token is set (GPT-2 does not have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to CPU (PyTorch quantization works best on CPU)
    model.to("mps")
    model.eval()  # Set to evaluation mode before quantization

    # Apply dynamic quantization (Reduces size and improves CPU inference speed)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )

    # Save the quantized model
    torch.save(quantized_model.state_dict(), quantized_model_path)

    print(f"Quantized model saved at: {quantized_model_path}")
