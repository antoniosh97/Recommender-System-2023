import torch
import numpy as np

class NCF(torch.nn.Module):
    def __init__(self, 
                 field_dims: list,
                 embed_dim: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = torch.nn.Embedding(field_dims[-1], embed_dim)
        sequential = [[self.embed_dim*2, 32], [32, 16], [16, 8]]
        self.mlp = torch.nn.Sequential(
                        torch.nn.Linear(sequential[0][0], sequential[0][1]), #  (64, 32)
                        torch.nn.ReLU(),
                        torch.nn.Linear(sequential[1][0], sequential[1][1]),
                        torch.nn.ReLU(),
                        torch.nn.Linear(sequential[2][0], sequential[2][1]),
                        torch.nn.ReLU())
        self.last_fc = torch.nn.Linear(sequential[2][1]+self.embed_dim, 1)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, interaction_pairs: np.ndarray) -> torch.Tensor:
        user_embedding_mf = self.embedding(interaction_pairs[:,0])
        item_embedding_mf = self.embedding(interaction_pairs[:,1])

        # MLP vector reshaped to 2 dimensions
        mlp_vector = self.embedding(interaction_pairs)  # (6142, 2, 32)
        mlp_vector = mlp_vector.reshape(-1, self.embed_dim*2) # (6142, 64)

        # MatrixFactorization vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf) # ( _ , 32)

        # MLP thought layers
        mlp_vector = self.mlp(mlp_vector) # ( _ , sequential[2][1]) = ( _ , 8)

        # NCF: concat MLP_output and MF_output
        # NCF: Last fully connected layer of size (40x1) --> (8: mlp output + 32: mf output => 40)
        # NCF: Activation function ReLU
        concatenation = torch.cat([mlp_vector, mf_vector], dim=-1) # (_, 40)
        concatenation = self.last_fc(concatenation)
        # concatenation = self.relu(concatenation)
        return concatenation.squeeze()

    def predict(self,
                interactions: np.ndarray,
                device: torch.device) -> torch.Tensor:
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores