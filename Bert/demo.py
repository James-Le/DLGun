from utils import *

# Params for bert model and tokenization
bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
max_seq_length = 256

train_df, test_df = download_and_load_datasets()

# Create datasets (Only take up to max_seq_length words for memory)
train_text = train_df["sentence"].tolist()
train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = train_df["polarity"].tolist()

test_text = test_df["sentence"].tolist()
test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = test_df["polarity"].tolist()

# Instantiate tokenizer
tokenizer = create_tokenizer_from_hub_module(bert_path)

# Convert data to InputExample format
train_examples = convert_text_to_examples(train_text, train_label)
test_examples = convert_text_to_examples(test_text, test_label)

# Convert to features
(
    train_input_ids,
    train_input_masks,
    train_segment_ids,
    train_labels,
) = convert_examples_to_features(
    tokenizer, train_examples, max_seq_length=max_seq_length
)
(
    test_input_ids,
    test_input_masks,
    test_segment_ids,
    test_labels,
) = convert_examples_to_features(
    tokenizer, test_examples, max_seq_length=max_seq_length
)

model = build_model(max_seq_length)

# Instantiate variables
initialize_vars(sess)

model.fit(
    [train_input_ids, train_input_masks, train_segment_ids],
    train_labels,
    validation_data=(
        [test_input_ids, test_input_masks, test_segment_ids],
        test_labels,
    ),
    epochs=1,
    batch_size=32,
)

