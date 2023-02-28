import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size, hidden_size):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.fc1 = nn.Linear(emb_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, user_id, item_id):
        user_vec = self.user_emb(user_id)
        item_vec = self.item_emb(item_id)
        x = torch.cat([user_vec, item_vec], dim=1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.Sigmoid()(x)
        return x