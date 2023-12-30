
import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertTokenizer
import warnings
warnings.filterwarnings('ignore')



class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 8
    EPOCHS = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BERT_PATH = "bert-base-uncased"
    MODEL_PATH = "bert-base-uncased/pytorch_model.bin"
    TRAINING_FILE = "../../Datasets/embold_train_cleaned.json"
    TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
    truncation = True



class BugPredictor(nn.Module):
    
    def __init__(self, n_classes):
        super(BugPredictor, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert_model.config.hidden_size, n_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(
        input_ids=input_ids,
        attention_mask = attention_mask
        )
        output = self.dropout(outputs[1])
        return self.out(output)
    


def predict_git_category(sample_message, model):
    encoded_message = config.TOKENIZER.encode_plus(sample_message, max_length=config.MAX_LEN,truncation=True, add_special_tokens=True, return_token_type_ids=False, pad_to_max_length=True, return_attention_mask=True, return_tensors='pt')
    input_ids = encoded_message['input_ids'].to(config.DEVICE)
    attention_mask = encoded_message['attention_mask'].to(config.DEVICE)
    
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    _, prediction_idx = torch.max(output, dim=1)
    class_names = ['bug', 'feature', 'question']
      
    return class_names[prediction_idx]

    
class_names = [0, 1, 2]
bug_predictor_model = BugPredictor(len(class_names)).to(config.DEVICE)
bug_predictor_model.load_state_dict(torch.load('best_model.bin'))

while True:
    print('Enter "exit" to exit')
    sample_message = input('Enter the issue text: ')
    if sample_message == 'exit':
        break
