import pandas as pd


# Converts the given list of transcripts into a list-of-lists, where
# each item is a list of "lines" (e.g. "Interviewee: blah blah")
# corresponding to a single transcript. If num_lines is not none, then
# the list will be truncated and padded (with empty sentences "") as necessary
# so that each script has the same number of lines post-processing.
def split_all_scripts(transcript_list, num_lines):
    all_split_scripts = []
    for script in transcript_list:
        lines = script.split("|")
        if num_lines is not None:
            if len(lines) > num_lines:
                lines = lines[0:num_lines]
        all_split_scripts.append(lines)

    for lines in all_split_scripts:
        while len(lines) < num_lines:
            lines.append("")
    return all_split_scripts


# Reads the transcripts from the CSV and processes them into split scripts.
def load_split_scripts(num_lines):
    transcripts_df = pd.read_csv("transcripts.csv", names=["Participant", "transcript"])
    return split_all_scripts(transcripts_df["transcript"], num_lines)
