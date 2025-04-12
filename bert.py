import itertools
from transformers import BertTokenizer, BertModel
import torch
import splits

# Smallest possible BERT model from Google.
# See https://huggingface.co/google/bert_uncased_L-2_H-128_A-2
# @article{turc2019,
#   title={Well-Read Students Learn Better: On the Importance of Pre-training Compact Models},
#   author={Turc, Iulia and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
#   journal={arXiv preprint arXiv:1908.08962v2 },
#   year={2019}
# }
default_model_name = "google/bert_uncased_L-2_H-128_A-2"

# --- SECTION: Tokenization ---


def default_tokenizer():
    return BertTokenizer.from_pretrained(default_model_name)


def tokenization_options():
    return {
        "padding": "max_length",
        # Some lines do have >512 tokens, and 512 is the maximum size for BERT input.
        "max_length": 512,
        "truncation": True,
        "return_tensors": "pt",
        "is_split_into_words": True,
    }


def tokenize(tokenizer, input):
    return tokenizer(input, **tokenization_options())


# Converts the given script into a tokenized representation.
def tokenize_split_script_batched(tokenizer, split_script):
    # return tokenize(tokenizer, split_script)
    return tokenizer.batch_encode_plus(split_script, **tokenization_options())


# Converts each line of the given script into a tokenized representation.
def tokenize_split_script_no_batch(tokenizer, split_script):
    return [tokenize(tokenizer, line) for line in split_script]


# Tokenizes all scripts in the given list, and confirms that the resulting
# output all has the same shape.
def tokenize_all_scripts(tokenizer, split_scripts):
    tokenized_scripts = [
        tokenize_split_script_batched(tokenizer, split_script)
        for split_script in split_scripts
    ]
    check_input_id_shape(tokenized_scripts)
    return tokenized_scripts


# Tokenizes each script line-by-line and flattens the output by concatenating the tokens.
def tokenize_all_lines(tokenizer, split_scripts):
    tokenized_scripts_lines = []
    # Manually flatten here via for-loops (instead of itertools) for clarity.
    for split_script in split_scripts:
        flattened_lines = []
        list_of_lines = tokenize_split_script_no_batch(tokenizer, split_script)
        for line in list_of_lines:
            flattened_lines.append(line)
        tokenized_scripts_lines.append(flattened_lines)
    flattened_tokenized_lines = list(
        itertools.chain.from_iterable(tokenized_scripts_lines)
    )
    check_input_id_shape(flattened_tokenized_lines)
    return tokenized_scripts_lines


def check_input_id_shape(tokenized_objects):
    tokenized_output_shape = None
    for token_output in tokenized_objects:
        if tokenized_output_shape is None:
            tokenized_output_shape = token_output["input_ids"].shape
        if token_output["input_ids"].shape != tokenized_output_shape:
            print(
                "Expected: ",
                tokenized_output_shape,
                " got: ",
                token_output["input_ids"].shape,
            )
        assert token_output["input_ids"].shape == tokenized_output_shape


# --- SECTION: Embeddings ---


# Executes the last phase of the embedding pipeline, converting a list
# of tokenized chunks of text into a list of embeddings.
def embeddings_for_tokenized_chunks(model, tokenized_chunks):
    embeddings = [
        # Run inference and get "pooler" output (whole-sequence embedding).
        model(**tokenized_chunk).pooler_output
        for tokenized_chunk in tokenized_chunks
    ]
    embedding_shape = None
    for embedding in embeddings:
        if embedding_shape is None:
            embedding_shape = embedding.shape
        assert embedding.shape == embedding_shape
    return embeddings


# Gets or computes the bert embeddings. If cache_eligible is True, then
# this function may read the embeddings off of the local disk.
def embeddings_for_transcripts(cache_eligible):
    cache_location = "bert_embeddings.pt"
    model_name = default_model_name
    if cache_eligible:
        try:
            return torch.load(cache_location)
        except Exception as e:
            print(f"Load error: {e}")

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # The max_num_lines was set by looking at the 95th percentile of number
    # of lines across the dataset. We are ok to ignore exceptionally long
    # interviews based on number of lines, although it could be an interesting
    # feature to add to the regressor.
    max_num_lines = 50
    split_scripts = splits.load_split_scripts(max_num_lines)
    batched = True
    embeddings = None
    if batched:
        embeddings = embeddings_for_tokenized_chunks(
            model, tokenize_all_scripts(tokenizer, split_scripts)
        )
    else:
        embeddings = [
            embeddings_for_tokenized_chunks(model, tokenized_line)
            for tokenized_line in tokenize_all_lines(tokenizer, split_scripts)
        ]
    if embeddings is None:
        print("Somehow failed to generate embeddings!")
        raise Exception

    # Concatenate the embeddings for each line, if necessary.
    flattened_embeddings = [torch.flatten(embedding) for embedding in embeddings]

    torch.save(flattened_embeddings, cache_location)
    return flattened_embeddings


### --- SECTION: Statistics and exploration ---


# Tokenizes the given split script with no truncation. This can be used
# to get statistics about the number of tokens in each line.
def tokenize_no_truncation(tokenizer, split_script):
    return tokenizer.batch_encode_plus(
        split_script,
        padding="longest",
        truncation=False,
        return_tensors="pt",
        is_split_into_words=True,
    )


# Tokenizes the given split scripts with no truncation.
def get_tokens(split_scripts):
    tokenizer = BertTokenizer.from_pretrained(default_model_name)
    return [
        tokenize_no_truncation(tokenizer, split_script)
        for split_script in split_scripts
    ]
