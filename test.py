import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
import processingdata as pdt  # Giả sử module này cung cấp từ điển vocab

# Đọc dữ liệu intents từ file bên ngoài (chỉnh sửa đường dẫn file cho phù hợp)
with open('data.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Lấy từ điển từ module processingdata (đảm bảo vocab có các mapping: word2idx, idx2word, '<unk>', '<pad>')
vocab = pdt.get_vocab()

# Tạo ánh xạ tag sang chỉ số
all_tags = [intent["tag"] for intent in intents["intents"]]
tag2idx = {tag: idx for idx, tag in enumerate(all_tags)}
idx2tag = {idx: tag for tag, idx in tag2idx.items()}

# Xây dựng Dataset cho bài toán phân loại intent
class IntentDataset(Dataset):
    def __init__(self, intents, vocab, tag2idx):
        self.samples = []
        for intent in intents['intents']:
            tag = intent['tag']
            for pattern in intent['patterns']:
                # Token hóa câu mẫu (ở đây đơn giản dùng split, có thể thay bằng thư viện khác nếu cần)
                tokens = pattern.split()
                # Chuyển đổi từ sang chỉ số, nếu từ không có trong vocab thì dùng token '<unk>'
                indices = [vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in tokens]
                self.samples.append((indices, tag2idx[tag]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Hàm collate để pad các câu về cùng độ dài trong 1 batch
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    max_len = max(lengths)
    # Sử dụng token '<pad>' để pad (nếu không có, có thể đặt giá trị 0)
    pad_idx = vocab.word2idx.get('<pad>', 0)
    padded_seqs = [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]
    return torch.tensor(padded_seqs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# Các siêu tham số
embedding_dim = 128
hidden_size = 256
num_layers = 1
batch_size = 16
epochs = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Định nghĩa mô hình classifier cho intent
class IntentClassifier(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x có kích thước [batch_size, seq_length]
        embedded = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        output, (hidden, cell) = self.lstm(embedded)
        # Lấy hidden state của layer cuối để phân loại
        logits = self.fc(hidden[-1])  # [batch_size, output_size]
        return logits

input_size = len(vocab)            # Số từ trong từ điển
output_size = len(tag2idx)         # Số lượng intent
model = IntentClassifier(input_size, embedding_dim, hidden_size, output_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Tạo DataLoader cho tập dữ liệu
dataset = IntentDataset(intents, vocab, tag2idx)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Huấn luyện mô hình
for epoch in range(epochs):
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch: [{epoch+1}/{epochs}], Loss: {loss.item():.3f}')

# Hàm dự đoán intent cho 1 câu đầu vào
def predict_intent(model, sentence, vocab):
    model.eval()
    tokens = sentence.split()
    indices = [vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in tokens]
    sentence_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(sentence_tensor)
        predicted_idx = logits.argmax(dim=1).item()
    return predicted_idx

# Vòng lặp chat: Dự đoán intent và trả về phản hồi ngẫu nhiên dựa trên dữ liệu intents
print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == 'exit':
        break
    pred_idx = predict_intent(model, user_input, vocab)
    pred_tag = idx2tag[pred_idx]
    # Tìm intent có tag trùng với dự đoán và chọn ngẫu nhiên 1 response
    for intent in intents['intents']:
        if intent['tag'] == pred_tag:
            response = random.choice(intent['responses'])
            break
    print("Chatbot:", response)
