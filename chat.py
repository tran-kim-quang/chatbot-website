import torch
from seq2seqModel import Seq2Seq
from processingdata import get_vocab
vocab = get_vocab()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(model, sentence, vocab, max_length=20):
    model.eval()
    sentence_tensor = torch.tensor([vocab.word2idx.get(word, vocab.word2idx['<unk>']) for word in sentence.split()], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(sentence_tensor)
        x = torch.tensor([vocab.word2idx['<sos>']], dtype=torch.long).to(device)
        response = []
        for _ in range(max_length):
            output, hidden, cell = model.decoder(x, hidden, cell, encoder_outputs)
            predicted_idx = output.argmax(1).item()
            if predicted_idx == vocab.word2idx['<eos>']:
                break
            response.append(vocab.idx2word[predicted_idx])
            x = torch.tensor([predicted_idx], dtype=torch.long).to(device)
    return ' '.join(response)

# Chạy chatbot
print("Chatbot đã sẵn sàng! Gõ 'exit' để thoát.")
while True:
    user_input = input("Bạn: ")
    if user_input.lower() == 'exit':
        break
    response = predict(model, user_input, vocab)
    print("Chatbot:", response)