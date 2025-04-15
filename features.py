import bert
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


def row_for_participant_id(participant_id):
    transcript_df = pd.read_csv("transcripts.csv", names=["Participant", "transcript"])
    return transcript_df.loc[transcript_df["Participant"] == participant_id].index[0]


def full_bert_features(participant_id):
    row = row_for_participant_id(participant_id)
    # print("Row :", row)
    all_bert_embeddings = bert.embeddings_for_transcripts(True)
    row_embedding = all_bert_embeddings[row]
    # print("Shape: ", row_embedding.shape)
    return row_embedding


def all_features(participant_id, only_interpretable):
    return full_bert_features(participant_id) if not only_interpretable else []


# Gets the features for all given participants as a tensor dataset whose first dimension
# is the index of the participant in all_participant_ids.
def feature_dataset_for_participants(all_participant_ids, only_interpretable):
    all_features_all_participants = [
        all_features(participant_id, only_interpretable)
        for participant_id in all_participant_ids
    ]
    return TensorDataset(torch.stack(all_features_all_participants))


# Gets the outcomes for all given participants as a tensor dataset whose first dimension
# is the index of the participant in all_participant_ids.
def outcome_dataset_for_participants(all_participant_ids):
    outcome_df = pd.read_csv("scores.csv", names=["Participant", "Overall", "Excited"])
    participant_rows = outcome_df.loc[
        outcome_df["Participant"].isin(all_participant_ids)
    ]
    # Drop the participant column.
    only_scores = participant_rows.drop(columns="Participant")
    return TensorDataset(torch.from_numpy(only_scores.to_numpy(dtype=float)))
