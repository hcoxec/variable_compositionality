from io import open
import unicodedata
import string
import re
import random
import spacy
import attr
import os
import json
from datetime import datetime


from collections import defaultdict
from tqdm import tqdm
from os.path import exists

import dataloader
from utils import registry

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Preprocesser():
    def __init__(self, args, split, add_EOS=False):
        
        self.args = args
        self.name = args.dataset
        self.split = split
        self.EOS_Token = 1
        self.SOS_Token = 0

        self.add_EOS = add_EOS
        
        in_lang, out_lang, pairs = self.prepareData(self.name, self.split)
        self.input_lang = in_lang 
        self.output_lang = out_lang
        self.raw_data = pairs
        self.tensor_x, self.tensor_y = self.tensorsFromPair()
        
    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence.split(' ')]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)
        if self.add_EOS:
            indexes.append(self.EOS_Token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
    
    def get_data(self, x_data, y_data, bs=32):
        train_ds = Dataset(x_data, y_data)
        train_dl = DataLoader(train_ds, batch_size=bs)
        return train_dl

    def tensorsFromPair(self):
        tensor_x, tensor_y = [], []
        for pair in self.raw_data:
            tensor_x.append(self.tensorFromSentence(self.input_lang, pair[0]))
            tensor_y.append(self.tensorFromSentence(self.output_lang, pair[1]))

        return tensor_x, tensor_y

    def readLangs(self, lang1, lang2, reverse=False):
        print(f"Reading lines from data/{self.name}/{self.split}.tsv ...")

        # Read the file and split into lines
        lines = open(f"data/{self.name}/{self.split}.tsv", encoding='utf-8').\
            read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [l.split('\t') for l in lines]

        Lang = registry.lookup("preprocess_lang", self.args.dataset)

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.args, lang2)
            output_lang = Lang(self.args, lang1)
        else:
            input_lang = Lang(self.args, lang1)
            output_lang = Lang(self.args, lang2)

        return input_lang, output_lang, pairs

    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = self.readLangs('source', 'target', reverse)
        print("Read %s sentence pairs" % len(pairs))
        for pair in tqdm(pairs, desc='Processing Data Pairs:'):
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

        input_lang.get_pos(input_lang.pos_words)
        output_lang.get_pos(output_lang.pos_words)

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
        return input_lang, output_lang, pairs
    
    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad, yy_pad, x_lens, y_lens

    def save_preprocess(self):
        save_path = f'data/{self.name}/preprocess/'
        for section in [
                ('enc_preprocess', self.input_lang), 
                ('dec_preprocess', self.output_lang)
                ]:

            section_path = f"{save_path}/{self.args.preprocess_version}_{self.split}/{section[0]}"
            os.makedirs(section_path, exist_ok= True)
            
            lang_dict = section[1].to_dict()
            with open(f"{section_path}/lang.json", "w") as outfile:
                json.dump(lang_dict, outfile, indent=4)

            with open(f"{section_path}/data.json", "w") as outfile:
                for sentence in section[1].sentences:
                    json.dump(sentence, outfile)
                    outfile.write('\n')


    
    def __len__(self):
        return len(self.raw_data)
    
    def __getitem__(self, idx):
        return self.tensor_x[idx], self.tensor_y[idx]
    
    def get_packed_loader(self, shuffle=True, bs=32):
        return DataLoader(self, batch_size=bs, collate_fn=self.pad_collate, shuffle=shuffle)


class TwoPredicatesCondenser():
    '''
    This is an artifact of earlier version of the MCD split code - essentially
    each split was processed separately, but that meant that the dataset object
    for each split had slightly different 'Langs' for mapping tokens to idx.
    This condenses across an arbitrary number of langs and makes their mappings
    identical

    '''
    def __init__(self) -> None:
        self.init = True

    def condense_langs(self, langs):
        base_lang = langs[0]

        for lang in langs[1:]:
            for word in lang.words:
                if word not in base_lang.words:
                    base_lang.addWord(word)

            for pos in lang.pos_words:
                for word in lang.pos_words[pos]:
                    base_lang.pos_words[pos].add(word)

            for dep in lang.dep_words:
                for word in lang.dep_words[dep]:
                    base_lang.dep_words[dep].add(word)

        return base_lang
    
    def reset_langs(
        self,
        condensed_input_lang, 
        condensed_output_lang, 
        all_preproc
    ):
        cond_in = condensed_input_lang.to_dict()
        cond_out = condensed_output_lang.to_dict()

        for split in all_preproc:
            split.input_lang.load(cond_in)
            split.output_lang.load(cond_out)
        
        return all_preproc

    def condense(self, all_preproc):
        input_dicts = [preproc.input_lang for preproc in all_preproc]
        output_dicts = [preproc.output_lang for preproc in all_preproc]
        condensed_input = self.condense_langs(input_dicts)
        condensed_output = self.condense_langs(output_dicts)

        return self.reset_langs(condensed_input, condensed_output, all_preproc)
    
def main(args):
    
    print(f"Beginning Preprocess at: {datetime.now()}")
    print("===================================================")


    all_preproc = [Preprocesser(args, split) for split in args.all_splits]
    condenser = TwoPredicatesCondenser()
    condensed_preproc = condenser.condense(all_preproc)

    for split in condensed_preproc:
        split.save_preprocess()
    