import os
import torch
import re

def get_embeddings(participant, feature_type, performance_type, k, root_dir_path:str ='/content/drive/MyDrive'):
  valid_feature_types = ['prosodic', 'sentiment']
  valid_performance_types = ['overall', 'excited']

  if not isinstance(participant, str) or not re.match(r'^([a-zA-Z]+[0-9]+)', participant):
      raise ValueError(f"Invalid participant ID '{participant}'. Must be a string like 'p1', 'pp2', etc.")

  if feature_type not in valid_feature_types:
      raise ValueError(f"Invalid feature_type '{feature_type}'. Must be one of {valid_feature_types}.")

  if performance_type not in valid_performance_types:
      raise ValueError(f"Invalid performance_type '{performance_type}'. Must be one of {valid_performance_types}.")

  if not isinstance(k, int) or k <= 0:
      raise ValueError(f"Invalid value for k: {k}. Must be a positive integer.")

  return torch.load(os.path.join(root_dir_path, f'data/{feature_type}/{performance_type}/{k}/{participant}.pt'))
