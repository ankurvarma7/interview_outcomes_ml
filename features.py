import bert
import EmbeddingGetter
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset

# Pre-load the dataframes at a module-level.
transcript_df = pd.read_csv("transcripts.csv", names=["Participant", "transcript"])
outcome_df = pd.read_csv("scores.csv", names=["Participant", "Overall", "Excited"])

# variance = outcome_df.iloc[1:]["Excited"].astype("float").mean()
# print(variance)


def row_index_for_participant_id(participant_id):
    return transcript_df.loc[transcript_df["Participant"] == participant_id].index[0]


def full_bert_features(participant_id, summary):
    row_index = row_index_for_participant_id(participant_id)
    # print("Row :", row_index)
    all_bert_embeddings = bert.embeddings_for_transcripts(True, summary)
    row_embedding = all_bert_embeddings[row_index]
    # print("Shape: ", row_embedding.shape)
    return row_embedding


def all_features(
    participant_id, include_prosodic, include_other, summarize_bert_features, k
):
    bert_features = full_bert_features(participant_id, summarize_bert_features)
    overall_sentiment_features = EmbeddingGetter.get_embeddings(
        participant_id, "sentiment", "overall", k, "."
    )
    overall_prosodic_features = EmbeddingGetter.get_embeddings(
        participant_id, "prosodic", "overall", k, "."
    )
    tensors_to_use = []
    if include_other:
        tensors_to_use.append(bert_features)
        tensors_to_use.append(overall_sentiment_features)
    if include_prosodic:
        tensors_to_use.append(overall_prosodic_features)
    return torch.cat(tensors_to_use, dim=0)


# Gets the outcomes for all given participants as a tensor dataset whose first dimension
# is the index of the participant in all_participant_ids.
def outcomes_for_participants(all_participant_ids):
    participant_rows = outcome_df.loc[
        outcome_df["Participant"].isin(all_participant_ids)
    ]
    # Drop the participant column.
    only_scores = participant_rows.drop(columns="Participant")
    return torch.from_numpy(only_scores.to_numpy(dtype=np.float32))


# Gets the features for all given participants as a tensor dataset whose first dimension
# is the index of the participant in all_participant_ids.
def dataset_for_participants(
    all_participant_ids,
    include_prosodic,
    include_other_features,
    summarize_bert_features,
    k,
):
    all_features_all_participants = [
        all_features(
            participant_id,
            include_prosodic,
            include_other_features,
            summarize_bert_features,
            k,
        )
        for participant_id in all_participant_ids
    ]
    return TensorDataset(
        torch.stack(all_features_all_participants),
        outcomes_for_participants(all_participant_ids),
    )
