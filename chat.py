import random
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load model parameters from file
MODEL_FILE = "data1.pth"
model_data = torch.load(MODEL_FILE)

input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data['all_words']
tags = model_data['tags']
responses_dict = model_data["responses"]
model_state = model_data["model_state"]

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ONGC"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device).float()

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        return f"{random.choice(responses_dict[tag])} (Probability: {prob.item():.2f})"
    return "I do not understand... (Probability: Low)"

if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break

        response = get_response(sentence)
        print(f"{bot_name}: {response}")
