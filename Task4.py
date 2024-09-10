import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, data_frame, tokenizer, max_length=512, is_test=False):
        self.data_frame = data_frame
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        row = self.data_frame.iloc[index]
        text = row['title'] + " " + row['sentence']
        inputs = self.tokenizer(text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors="pt")
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)

        if self.is_test:
            return input_ids, attention_mask
        else:
            score = torch.tensor(row['score']).float()
            return input_ids, attention_mask, score

class MyModule(nn.Module):
    def __init__(self, bert_model):
        super(MyModule, self).__init__()
        self.bert = bert_model
        self.regressor = nn.Linear(768, 1) 

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        hidden_state = outputs.pooler_output
        score = self.regressor(hidden_state)
        return score

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    batch = 4
    epochs = 10

    train_val = pd.read_csv("train.csv")
    test_val = pd.read_csv("test_no_score.csv")

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    train_dataset = ReviewDataset(train_val, tokenizer, is_test=False)
    test_dataset = ReviewDataset(test_val, tokenizer, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=4)

    model = MyModule(bert_model).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        for input_ids, attention_mask, scores in tqdm(train_loader, total=len(train_loader)):
            input_ids, attention_mask, scores = input_ids.to(device), attention_mask.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, scores.unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    results = []
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(test_loader, total=len(test_loader)):
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask)
            results.append(outputs.cpu().numpy())

    results = np.concatenate(results).flatten()
    with open("result.txt", "w") as f:
        for val in results:
            f.write(f"{val}\n")

if __name__ == '__main__':
    main()
