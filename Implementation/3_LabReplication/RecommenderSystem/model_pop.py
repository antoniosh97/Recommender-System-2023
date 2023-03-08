import torch
import numpy as np

class PopularityBasedModel(torch.nn.Module):
  """
  https://github.com/LaxmiChaudhary/Amzon-Product-Recommendation/blob/master/Recommendation%20System.ipynb
  The Popularity-based recommender system is a non-personalised recommender system and these are based on frequecy counts, which may be not suitable to the user.
  We can see the differance above for the user id 4, 6 & 8, The Popularity based model has recommended the same set of 5 products to both.
  """
  def __init__(self, 
                 popularity_recommendations) -> None:
        super(PopularityBasedModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.popularity_recommendations = popularity_recommendations
  
  def predict(self,
              interactions: np.ndarray,
              device=None) -> torch.Tensor:
      return torch.IntTensor(self.popularity_recommendations)