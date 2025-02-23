import processingdata as pdt
from encoder import Encoder
import json
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Load dữ liệu intents
with open('data.json', 'r', encoding='utf-8') as f:
    intent_data = json.load(f)
    intents = intent_data['intents']
    vocab = pdt.get_vocab()

# Xây dựng tập dữ liệu huấn luyện cho phân loại intent
train_data = []
intent_labels = {intent['tag']: i for i, intent in enumerate(intents)}
label_to_intent = {i: tag for tag, i in intent_labels.items()}

for intent in intents:
    for pattern in intent['patterns']:
        pattern_tensor = [vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in pattern.split()]
        train_data.append((pattern_tensor, intent_labels[intent['tag']]))

# Khởi tạo mô hình phân loại
input_size = len(vocab)
num_classes = len(intent_labels)
hidden_size = 128
embedding_dim = 64

class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dim, num_classes):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.rnn(embedded)
        output = self.fc(hidden[-1])
        return output

# Thiết lập thiết bị và huấn luyện mô hình
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IntentClassifier(input_size, hidden_size, embedding_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for sentence, label in train_data:
        sentence_tensor = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(device)
        label_tensor = torch.tensor([label], dtype=torch.long).to(device)
        
        output = model(sentence_tensor)
        loss = criterion(output, label_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_data):.4f}')

# Hàm dự đoán intent và chọn phản hồi ngẫu nhiên
def predict_intent(sentence):
    model.eval()
    sentence_tensor = torch.tensor([vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in sentence.split()], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(sentence_tensor)
        predicted_label = torch.argmax(output, dim=1).item()
    intent_tag = label_to_intent[predicted_label]
    for intent in intents:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    return "Xin lỗi, tôi không hiểu."

# Chạy chatbot
print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == 'exit':
        break
    response = predict_intent(user_input)
    print("Chatbot:", response)
