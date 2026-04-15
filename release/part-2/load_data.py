import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.is_test = split == "test"
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.decoder_start_id = self.tokenizer.pad_token_id
        self.encoder_ids, self.decoder_inputs, self.decoder_targets, self.initial_decoder_inputs = self.process_data(
            data_folder, split, self.tokenizer
        )

    def process_data(self, data_folder, split, tokenizer):
        input_lines = load_lines(os.path.join(data_folder, f"{split}.nl"))
        encoder_ids = []
        decoder_inputs = []
        decoder_targets = []
        initial_decoder_inputs = []

        if split != "test":
            output_lines = load_lines(os.path.join(data_folder, f"{split}.sql"))
        else:
            output_lines = None

        for idx, nl_text in enumerate(input_lines):
            source_text = f"translate to SQL: {nl_text}"
            source_ids = tokenizer(source_text, add_special_tokens=True).input_ids
            source_ids = torch.tensor(source_ids, dtype=torch.long)
            encoder_ids.append(source_ids)

            initial_decoder_inputs.append(torch.tensor([self.decoder_start_id], dtype=torch.long))

            if output_lines is not None:
                target_ids = tokenizer(output_lines[idx], add_special_tokens=True).input_ids
                target_ids = torch.tensor(target_ids, dtype=torch.long)

                shifted_input = torch.cat(
                    [
                        torch.tensor([self.decoder_start_id], dtype=torch.long),
                        target_ids[:-1],
                    ]
                )

                decoder_inputs.append(shifted_input)
                decoder_targets.append(target_ids)

        return encoder_ids, decoder_inputs, decoder_targets, initial_decoder_inputs
    
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        if self.is_test:
            return self.encoder_ids[idx], self.initial_decoder_inputs[idx]
        return (
            self.encoder_ids[idx],
            self.decoder_inputs[idx],
            self.decoder_targets[idx],
            self.initial_decoder_inputs[idx],
        )

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, decoder_inputs, decoder_targets, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, 'train.nl'))
    train_y = load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x = load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y = load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x = load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x