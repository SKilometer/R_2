import ast
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class PointwiseRankDataset(Dataset):
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
            for query, score in rewritten_questions:
                normalized_score = score / 1.0  # 假设分数最大为5，进行标准化
                pairs.append((query, original_question, context, normalized_score))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        query, original_question, context, label = self.pairs[idx]

        # 计算剩余的长度以保持整体不超长
        remaining_length = self.max_length - 4  # 3为[CLS], [SEP], [SEP]的长度
        len_query = len(self.tokenizer.tokenize(query))
        len_original = len(self.tokenizer.tokenize(original_question))
        len_context = remaining_length - len_query - len_original
        context_tokens = self.tokenizer.tokenize(context)[:len_context]
        context = self.tokenizer.convert_tokens_to_string(context_tokens)

        inputs = self.tokenizer(
            text=query,
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


class PointwiseRanker(nn.Module):
    def __init__(self, model):
        super(PointwiseRanker, self).__init__()
        self.bert = model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),  # hidden_size = 768
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        score = self.fc(cls_emb)
        return score


model_path = '../bert-base-uncased'
tokenizer_path = '../bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
model = BertModel.from_pretrained(model_path).to(device)
model.eval()

# 调用加载函数
training_data = pd.read_csv('../dataset/question_context_qr_score.csv')
# training_data = training_data.head(200)
dataset = PointwiseRankDataset(training_data, tokenizer)
print(f"Dataset size: {len(dataset)}")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

ranker = PointwiseRanker(model).to(device)
ranker = torch.nn.DataParallel(ranker)
# print("ranker:", ranker)
optimizer = optim.AdamW(ranker.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
loss_fn = nn.MSELoss()

num_epochs = 100
train_losses = []
test_losses = []

for epoch in tqdm(range(num_epochs)):
    ranker.train()
    train_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        scores = ranker(input_ids, attention_mask).squeeze(1)
        loss = loss_fn(scores, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()
    train_losses.append(train_loss / len(train_loader))

    ranker.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            scores = ranker(input_ids, attention_mask).squeeze(1)
            loss = loss_fn(scores, labels)
            test_loss += loss.item()

    test_losses.append(test_loss / len(test_loader))

    print(f'Epoch: {epoch}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {test_loss / len(test_loader)}')

model_dir = './pointwise_ranker'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()

loss_curve_path = os.path.join(model_dir, 'pointwise_loss_curve.png')
plt.savefig(loss_curve_path)  # 保存损失曲线图
plt.show()

model_save_path = os.path.join(model_dir, "pointwise_ranker.pth")
torch.save({
    'model_state_dict': ranker.module.state_dict(),
}, model_save_path)
print(f"Model saved to {model_save_path}")
