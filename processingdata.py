import json
import re
from collections import defaultdict
# Đọc dữ liệu từ file JSON
with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    text = text.lower()  # Chuyển thành chữ thường
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    return text

# Hàm token hóa văn bản
def tokenize(text):
    return text.split()

# Tạo từ điển
class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

# Khởi tạo từ điển và thêm các token đặc biệt
vocab = Vocabulary()
vocab.add_word('<pad>')  # Token padding
vocab.add_word('<sos>')  # Start of sentence
vocab.add_word('<eos>')  # End of sentence
vocab.add_word('<unk>')  # Unknown token

# Duyệt qua dữ liệu để xây dựng từ điển
for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern = preprocess_text(pattern)
        for word in tokenize(pattern):
            vocab.add_word(word)
    for response in intent['responses']:
        response = preprocess_text(response)
        for word in tokenize(response):
            vocab.add_word(word)

# Hàm chuyển câu thành tensor
def sentence_to_tensor(sentence, vocab):
    sentence = preprocess_text(sentence)
    tokens = tokenize(sentence)
    return [vocab.word2idx.get(token, vocab.word2idx['<unk>']) for token in tokens]

# Chuẩn bị dữ liệu huấn luyện
train_data = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        pattern_tensor = sentence_to_tensor(pattern, vocab)
        for response in intent['responses']:
            response_tensor = sentence_to_tensor(response, vocab)
            train_data.append((pattern_tensor, response_tensor))

# Thêm token <sos> và <eos> vào các câu
sos_token = vocab.word2idx['<sos>']
eos_token = vocab.word2idx['<eos>']

final_data = []
for pattern_tensor, response_tensor in train_data:
    pattern_tensor = [sos_token] + pattern_tensor + [eos_token]
    response_tensor = [sos_token] + response_tensor + [eos_token]
    final_data.append((pattern_tensor, response_tensor))

# In ra dữ liệu cuối cùng để kiểm tra
print("Dữ liệu cuối cùng:", final_data)

# Lưu từ điển và dữ liệu đã xử lý vào file JSON
data_to_save = {
    "vocab": vocab.word2idx,
    "data": final_data
}

with open('processed_intents.json', 'w', encoding='utf-8') as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)

print("Dữ liệu đã được lưu vào file 'processed_intents.json'")
def get_vocab():
    return vocab