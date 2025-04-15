import bert
import pandas as pd

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