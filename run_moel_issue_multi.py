import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./codebert_model")
tokenizer2 = BertTokenizer.from_pretrained("./bert_model")
labels = {'positive': 1, 'negative': 0}

class Multi_Issue_Dataset(Dataset):
    def __init__(self, dataframe):
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in dataframe['text']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts

class Multi_Issue_classifier(nn.Module):
    def __init__(self):
        super(Multi_Issue_classifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert_model")
        self.dropout_rate = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, text_id, text_mask):
        text_output = self.bert(input_ids=text_id, attention_mask=text_mask, return_dict=False)[0]
        text_cls = text_output[:, 0, :]
        hidden_input = text_cls
        hidden_output = self.hidden(hidden_input)
        return hidden_output

def run_model_issue_multi(dataset_PATH, best_model_path="parameters/issue_multi_best_model.pth"):
    model = Multi_Issue_classifier()
    df = pd.read_csv(dataset_PATH)
    dataset = Multi_Issue_Dataset(df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Now using {device}")

    print("Now loading the best model parameters...")
    try:
        model.load_state_dict(torch.load(best_model_path))
        print("Read best model successfully. Starting training...")
    except:
        raise Exception("No best model found.")

    model.to(device)

    if use_cuda:
        model = model.cuda()

    model.eval()
    all_preds = []

    with torch.no_grad():
        for texts in tqdm(dataloader):
            text_ids = texts['input_ids'].squeeze(1).to(device)
            attention_mask = texts['attention_mask'].squeeze(1).to(device)

            outputs = model(text_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())

    return all_preds

if __name__ == '__main__':
    all_preds = run_model_issue_multi("data/dataset_torvalds_linux_issue.csv")
    print(all_preds)
