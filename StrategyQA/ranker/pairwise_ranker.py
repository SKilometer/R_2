import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import ast
import os

"""
    Pairwise Ranker--(q1,q2,Q,C,label)
    预训练的bert为基座，训练一个排序器，输入（query+question）和label，返回score
    loss BCEWithLogits
"""

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Pairwise_RankerDataset(Dataset):
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
            context = row['Context']
            for i in range(len(rewritten_questions)):
                for j in range(i + 1, len(rewritten_questions)):
                    qi, score_i = rewritten_questions[i]
                    qj, score_j = rewritten_questions[j]
                    label = 1 if score_i >= score_j else 0
                    pairs.append((qi, qj, original_question, context, label))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query1, query2, original_question, context, label = self.pairs[idx]

        # 计算剩余的长度以保持整体不超长
        remaining_length = self.max_length - 5  # 3为[CLS], [SEP], [SEP]的长度
        len_query1 = len(self.tokenizer.tokenize(query1))
        len_query2 = len(self.tokenizer.tokenize(query2))
        len_original = len(self.tokenizer.tokenize(original_question))
        len_context = remaining_length - len_query1 - len_query2 - len_original
        context_tokens = self.tokenizer.tokenize(context)[:len_context]
        context = self.tokenizer.convert_tokens_to_string(context_tokens)

        inputs = self.tokenizer(
            text=query1 + "[SEP]" + query2,
            text_pair=original_question + "[SEP]" + context,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class Pairwise_Ranker(nn.Module):
    def __init__(self, model):
        super(Pairwise_Ranker, self).__init__()
        self.bert = model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),  # hidden_size = 768
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        diff = self.fc(cls_emb)
        return diff


model_path = '../bert-base-uncased'
tokenizer_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)
model.eval()

# 调用加载函数
training_data = pd.read_csv('../dataset/question_context_qr_score.csv')
# training_data = training_data.head(200)
dataset = Pairwise_RankerDataset(training_data, tokenizer)
print(f"Dataset size: {len(dataset)}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

ranker = Pairwise_Ranker(model).to(device)
ranker = torch.nn.DataParallel(ranker)
print(ranker)

optimizer = optim.AdamW(ranker.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 100
train_losses = []
val_losses = []

for epoch in tqdm(range(num_epochs)):
    ranker.train()
    train_loss = 0

    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)  # .unsqueeze(1)

        optimizer.zero_grad()
        diff = ranker(input_ids, attention_mask)
        loss = criterion(diff, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_losses.append(train_loss / len(train_loader))

    ranker.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)  # .unsqueeze(1)

            diff = ranker(input_ids, attention_mask)
            loss = criterion(diff, labels)
            val_loss += loss.item()

    val_losses.append(val_loss / len(test_loader))

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
loss_curve_path = os.path.join(model_dir, 'pairwise_loss_curve.png')
plt.savefig(loss_curve_path)  # 保存损失曲线图
plt.show()

model_save_path = os.path.join(model_dir, "pairwise.pth")
torch.save({
    'model_state_dict': ranker.module.state_dict(),  # 如果是 DataParallel 包装的模型
}, model_save_path)
print(f"Model saved to {model_save_path}")
