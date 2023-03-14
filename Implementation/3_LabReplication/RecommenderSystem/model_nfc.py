import torch
import numpy as np
import torch.nn as nn

class NCF(torch.nn.Module):
    def __init__(self, 
                 field_dims: list,
                 embed_dim: float) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_users = field_dims[0]
        self.num_items = (field_dims[1] - field_dims[0])     
        #self.num_items = (field_dims[1]) 

        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)

        self.embedding_user_mf = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embed_dim)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embed_dim)

        sequential = [[self.embed_dim*2, 32], [32, 16], [16, 8]]
        self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(sequential[0][0], sequential[0][1]), #  (64, 32)
                        torch.nn.ReLU(),
                        torch.nn.Linear(sequential[1][0], sequential[1][1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(sequential[2][0], sequential[2][1]),
                        torch.nn.ReLU())
        self.last_fc = torch.nn.Linear(sequential[2][1]+self.embed_dim, 1)
        # self.logistic = nn.Sigmoid()
        self.init_weight() 
    
    def init_weight(self):
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.last_fc.weight)

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, interaction_pairs: np.ndarray) -> torch.Tensor:

        user_embedding_mf = self.embedding_user_mf(interaction_pairs[:,0])
        item_embedding_mf = self.embedding_item_mf(interaction_pairs[:,1] - self.num_users)

        user_embedding_mlp = self.embedding_user_mlp(interaction_pairs[:,0])
        item_embedding_mlp = self.embedding_item_mlp(interaction_pairs[:,1] - self.num_users)

        # MLP vector reshaped to 2 dimensions
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector

        # MatrixFactorization vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf) # ( _ , 32)

        # MLP thought layers
        mlp_vector = self.mlp(mlp_vector) # ( _ , sequential[2][1]) = ( _ , 8)

        # NCF: concat MLP_output and MF_output
        # NCF: Last fully connected layer of size (40x1) --> (8: mlp output + 32: mf output => 40)

        concatenation = torch.cat([mlp_vector, mf_vector], dim=-1) # (_, 40)
        logits = self.last_fc(concatenation)
        # we modify the original code with Sigmoid here
        # if sigmoid is not included we obtain the logits from the las MLP
        # we return the logits and then apply the BCEwithLogitsLoss  
        # this is usually preferred due to numerical stability.

        return logits.squeeze()
        # rating = self.logistic(logits) 
        # return rating.squeeze()

    def predict(self,
                interactions: np.ndarray,
                device: torch.device) -> torch.Tensor:
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        ratings = self.forward(test_interactions)
        return ratings