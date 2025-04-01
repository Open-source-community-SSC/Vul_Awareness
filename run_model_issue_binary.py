import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
best_f1_path = "model_best/mode_issue_binary_f1.txt"
best_model_path = "model_best/model_issue_binary_model.pth"

# Load pre-trained tokenizers
tokenizer1 = AutoTokenizer.from_pretrained("./codebert_model")
tokenizer2 = BertTokenizer.from_pretrained("./bert_model")
labels = {'positive': 1, 'negative': 0}

class Binary_Issue_Dataset(Dataset):
    def __init__(self, dataframe):
        self.texts = [tokenizer2(text, padding='max_length', max_length=512, truncation=True, return_tensors="pt") for text in dataframe['text']]
        self.labels = [labels[label] for label in dataframe['flag']]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        return batch_texts

# Model
class Binary_Issue_Classifier(nn.Module):
    def __init__(self):
        super(Binary_Issue_Classifier, self).__init__()
        self.bert1 = AutoModelForSequenceClassification.from_pretrained("mycodebert")
        self.bert2 = BertModel.from_pretrained("mybert")
        self.dropout = 0.3
        self.hidden = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(8, 2),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )


    def forward(self, text, mask):
        pooled_output1 = self.bert2(input_ids=text, attention_mask=mask, return_dict=False)[0]
        output_text = pooled_output1[:, 0, :]
        # need new compare method
        output = self.hidden(output_text)
        return output

def run_model_issue_binary(dataset_PATH):
    model = Binary_Issue_Classifier()
    df = pd.read_csv(dataset_PATH)
    dataset = Binary_Issue_Dataset(df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Now using {device}")

    print("Now loading the best model parameters...")
    try:
        with open(best_f1_path, 'r') as f:
            best_f1_score = float(f.read())
            f.close()
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

