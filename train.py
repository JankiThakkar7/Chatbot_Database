import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words, load_data_from_database
from nltk.corpus import stopwords
import nltk

# Download stopwords from NLTK
# nltk.download('stopwords')

class ChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = X
        self.y_data = y
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

def preprocess_data(data):
    all_words = []
    tags = []
    xy = []
    
    stop_words = set(stopwords.words('english'))  # Load stopwords from NLTK

    for record in data:
        tag = record['tag']
        responses = record['responses']
        if tag not in tags:
            tags.append(tag)
        pattern = record['pattern']
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',', '--', '-', '(', ')', '/', '`']
    all_words = [stem(w) for w in all_words if w not in ignore_words and w not in stop_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    X = []
    y = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X.append(bag)
        label = tags.index(tag)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y, all_words, tags

def main():
    data = load_data_from_database()
    
    if data is None:
        print("Failed to load data from database. Exiting...")
        return

    X, y, all_words, tags = preprocess_data(data)

    # Hyperparameters
    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(X[0])
    learning_rate = 0.001
    num_epochs = 500

    dataset = ChatDataset(X, y)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i, (words, labels) in enumerate(train_loader):
            words = words.to(device).float()
            labels = labels.to(device).long()

            outputs = model(words)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss:.4f}')

    # Save responses for inference
    responses_dict = {tag: [] for tag in tags}
    for record in data:
        responses_dict[record['tag']] = record['responses']

    # Save the trained model and other required data
    model_data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "output_size": output_size,
        "hidden_size": hidden_size,
        "all_words": all_words,
        "tags": tags,
        "responses": responses_dict
    }

    FILE = "data.pth"
    torch.save(model_data, FILE)

    print(f'Training complete. Model saved to {FILE}')

if __name__ == "__main__":
    main()
