import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import ast
import os

"""
    Pairwise Ranker--RankNET (q1,q2,Q,C,label)->(q1,Q,C) and (q2,Q,C)
    预训练的bert为基座，训练一个排序器，输入（query+question）和label，返回score
    loss BCEWithLogits
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RankNetDataset(Dataset):
    def __init__(self, training_data, tokenizer, max_length=512):
        self.data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        for idx, row in self.data.iterrows():
            original_question = row['Original Question']
            rewritten_questions = ast.literal_eval(row['Rewritten Queries with Scores'])
            for i in range(len(rewritten_questions)):
                for j in range(i + 1, len(rewritten_questions)):
                    qi, score_i = rewritten_questions[i]
                    qj, score_j = rewritten_questions[j]
                    label = 1 if score_i >= score_j else 0
                    pairs.append((qi, qj, original_question, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def _encode_query(self, query, original_question):

        inputs = self.tokenizer(
            text=query,
            text_pair=original_question,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return inputs

    def __getitem__(self, idx):
        query1, query2, original_question, label = self.pairs[idx]

        encoded1 = self._encode_query(query1, original_question)
        encoded2 = self._encode_query(query2, original_question)

        return {
            'input_ids1': encoded1['input_ids'].squeeze(0),
            'attention_mask1': encoded1['attention_mask'].squeeze(0),
            'input_ids2': encoded2['input_ids'].squeeze(0),
            'attention_mask2': encoded2['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class RankNET(nn.Module):
    def __init__(self, model):
        super(RankNET, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(0.5)  # 新增Dropout层
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),  # hidden_size = 768
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def get_score(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = self.dropout(cls_emb)  # 应用Dropout
        score = self.fc(cls_emb)
        return score.squeeze(-1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        score1 = self.get_score(input_ids=input_ids1, attention_mask=attention_mask1)
        score2 = self.get_score(input_ids=input_ids2, attention_mask=attention_mask2)
        diff = score1 - score2
        return diff

    def predict(self, input_ids, attention_mask):
        return self.get_score(input_ids, attention_mask)


model_path = '../bert-base-uncased'
tokenizer_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)

# 调用加载函数
training_data = pd.read_csv('../dataset/question_context_qr_score.csv')
original_questions = training_data['Original Question'].unique()
train_orig, val_orig = train_test_split(original_questions, test_size=0.2, random_state=42)
train_data = training_data[training_data['Original Question'].isin(train_orig)].reset_index(drop=True)
val_data = training_data[training_data['Original Question'].isin(val_orig)].reset_index(drop=True)
train_dataset = RankNetDataset(train_data, tokenizer)
val_dataset = RankNetDataset(val_data, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

ranker = RankNET(model).to(device)
ranker = torch.nn.DataParallel(ranker)

optimizer = optim.AdamW([
    {'params': ranker.module.bert.parameters(), 'lr': 1e-6},
    {'params': ranker.module.fc.parameters(), 'lr': 1e-4}
], weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-7)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 5
counter = 0

for epoch in tqdm(range(num_epochs)):
    ranker.train()
    train_loss = 0

    for batch in train_loader:
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        labels = batch['label'].to(device)  # .unsqueeze(1)

        optimizer.zero_grad()
        diff = ranker(input_ids1, attention_mask1, input_ids2, attention_mask2)
        loss = criterion(diff, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(ranker.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_losses.append(train_loss / len(train_loader))

    ranker.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)  # .unsqueeze(1)

            diff = ranker(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = criterion(diff, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_loader))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        # 保存最佳模型
        torch.save({
            'model_state_dict': ranker.module.state_dict(),
        }, 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    print(f'Epoch {epoch + 1}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

model_dir = './pairwise_ranker'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
loss_curve_path = os.path.join(model_dir, 'RankNET_loss_curve.png')
plt.savefig(loss_curve_path)  # 保存损失曲线图
plt.show()

model_save_path = os.path.join(model_dir, "pairwise_RankNET.pth")
torch.save({
    'model_state_dict': ranker.module.state_dict(),  # 如果是 DataParallel 包装的模型
}, model_save_path)
print(f"Model saved to {model_save_path}")
