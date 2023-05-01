from sklearn.preprocessing import scale
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import attr
import re
import unicodedata
import json

from collections import defaultdict

from utils import registry


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@attr.s
class TwoPredicatesExample:
    source = attr.ib()
    target = attr.ib()
    dep_tags = attr.ib(default=None)
    pos_tags = attr.ib(default=None)
    source_tensor = attr.ib(default=None)
    target_tensor = attr.ib(default=None)

class Lang:
    ''''
    A general class for preprocessing datasets. A lang is created
    for the input and output languages. This should be subclassed for each
    dataset. Built-in methods for normalization can be overwritten.
    Each subclss should implement methods for adding each sentence to 
    the language
    
    '''
    def __init__(self, args, name):
        pass

    def addSentence(self, sentence):
        raise NotImplementedError

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?_()]+", r" ", s)
        return s

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.words.append(word)
        else:
            self.word2count[word] += 1

    def to_dict(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx] 


@registry.register("preprocess_lang", "two_predicates")
class TwoPredicatesLang(Lang):
    def __init__(self, args, name):
        self.args = args
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.eos_idx = 0
        self.words = []
        self.index2word = {2: "<SOS>", 1: "<EOS>", 0: "<PAD>", 3: "<UNK>"}
        self.n_words = 2  # Count SOS and EOS
        self.num_special = len(self.words)

        self.pos_words = defaultdict(lambda: set())
        self.dep_words = defaultdict(lambda: set())

        self.sentences = []

        self.pos_tags = []
        self.dep_tags = []

    def addSentence(self, sentence):
        sentence = self.normalizeString(sentence)
        dep, pos = ['subj', 'verb', 'obj'], ['NOUN', 'VERB', 'NOUN']

        for i, word in enumerate(sentence.split(' ')):
            self.addWord(word)
            self.pos_words[pos[i]].add(word)
            self.dep_words[dep[i]].add(word)
 
        self.sentences.append(
            {
                self.name : sentence.split(' '), 
                'pos_tags': pos, 
                'dep_tags': dep
            }
        )

    # Turn a Unicode string to plain ASCII, thanks to
    # https://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Lowercase, trim, and remove non-letter characters

    def normalizeString(self, s):
        s = self.unicodeToAscii(s.strip())
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?_()]+", r" ", s)
        return s

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
            self.words.append(word)
        else:
            self.word2count[word] += 1

    def to_dict(self):
        list_pos_words, list_dep_words = {}, {}
        for pos in self.pos_words.keys():
            list_pos_words[pos] = list(self.pos_words[pos])

        for dep in self.dep_words.keys():
            list_dep_words[dep] = list(self.dep_words[dep])

        lang_dict = {
            'name': self.name,
            'vocab': self.words,
            'word2index': self.word2index,
            'index2word': self.index2word,
            'word2count': self.word2count,
            'pos_words': list_pos_words,
            'dep_words': list_dep_words,
        }
        return lang_dict

    def __len__(self):
        return len(self.index2word.keys())+3 

    def __getitem__(self, idx):
        return self.words[idx] if idx < len(self.words) else self.unk_word


    def get_pos(self, pos_words):
        for pos in pos_words.keys():
            s_pos = pos.lower()+'s'
            setattr(self, s_pos, list(pos_words[pos]))

    def load(self, as_dict):
        self.words = as_dict['vocab']
        self.word2index = as_dict['word2index']
        self.index2word = as_dict['index2word']
        self.word2count = as_dict['word2count']
        self.pos_words = as_dict['pos_words']
        self.dep_words = as_dict['dep_words']
        self.n_words = len(self.index2word)

        self.get_pos(self.pos_words)



@registry.register("dataset", "two_predicates")
class TwoPredicatesDataset(Dataset):
    def __init__(self, args, split, scaling=None) -> None:
        self.args = args
        self.split = split

        if scaling:
            self.data_scaling = scaling
        elif hasattr(args, 'data_scaling'):
            self.data_scaling = args.data_scaling
        else:
            self.data_scaling = 1
                    
        self.load_preproc()
        self.tensors_from_preproc(self.enc_lang, self.dec_lang, self.examples)

        self.true_len = len(self.examples)

        self.n_attributes = len(self.enc_lang.dep_words.keys())
        self.n_values = self.enc_lang.n_words

        super().__init__()

    def load_preproc(self):
        save_path = f'data/{self.args.dataset}/preprocess'

        paired_examples, langs, loaded_data = [], [], []
        for section in ['enc_preprocess', 'dec_preprocess']:
            section_path = f"{save_path}/{self.args.preprocess_version}_{self.split}/{section}"
            with open(f"{section_path}/lang.json", "rb") as outfile:
                lang = json.load(outfile)
            examples = []
            with open(f"{section_path}/data.json") as outfile:
                for line in outfile:
                    examples.append(json.loads(line))
            
            langs.append(lang)
            paired_examples.append(examples)

        for src_example, target_example in zip(*paired_examples):
            ex = TwoPredicatesExample(
                source=src_example['source'],
                target=target_example['target'],
                dep_tags=src_example['dep_tags'],
                pos_tags=src_example['pos_tags']

            )
            loaded_data.append(ex)


        enc_lang, dec_lang = TwoPredicatesLang(self.args, 'encoder'), TwoPredicatesLang(self.args, 'decoder')
        enc_lang.load(langs[0])
        dec_lang.load(langs[1])

        self.enc_lang = enc_lang
        self.dec_lang = dec_lang
        self.examples = loaded_data

        print(f'   -{self.split} Preprocess found and loaded')

    def indexesFromSentence(self, lang, sentence):
        return [lang.word2index[word] for word in sentence]

    def tensorFromSentence(self, lang, sentence):
        indexes = self.indexesFromSentence(lang, sentence)

        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def tensorsFromPair(self, data_pairs):
        tensor_x, tensor_y = [], []
        for pair in data_pairs:
            tensor_x.append(self.tensorFromSentence(self.enc_lang, pair[0]))
            tensor_y.append(self.tensorFromSentence(self.dec_lang, pair[1]))

        return tensor_x, tensor_y

    def make_hot(self, example, n_words):
        one_hot = torch.zeros(len(example), n_words, dtype=torch.float)
        for w, word_idx in enumerate(example):
            one_hot[w][word_idx] = 1
            
        return one_hot

    def batched_make_hot(self, data, n_words):
        one_hots = []
        for example in data:
            this_grid = torch.zeros(len(example), n_words, dtype=torch.float)
            for w, word_idx in enumerate(example):
                this_grid[w][word_idx] = 1
            one_hots.append(this_grid.view(-1))
        
        return one_hots

    def build_idx_phrases(self):
        idx_phrases = []
        for i, phrase in enumerate(self.training_data.tensor_x):
            this_grid = torch.zeros(len(self.training_data.tensor_x))
            for w, word in enumerate(phrase):
                word_idx = self.inputword2index[word]
                this_grid[w]= word_idx
            idx_phrases.append(this_grid)
        return torch.stack(idx_phrases)

    def logits_to_idx(self, data_set, batched = False):
        '''
        creates index based reconstructions from logits of attributes and values

        inputs:

            dataset: default expects stack of logit tensors
            batched: if not stack then batched (not tested recently)
        '''
        
        idx_phrases = []

        if batched:
            for x, y in data_set:
                one_hot = x.view(x.shape[0],self.n_attributes, self.n_values)
                idx_phrase = torch.argmax(one_hot, dim=2)
                idx_phrases.extend(idx_phrase)

        else:
            one_hot = data_set.view(data_set.shape[0],self.n_attributes, self.n_values)
            return torch.argmax(one_hot, dim=2)
            #idx_phrases.append(idx_phrase)

        return torch.stack(idx_phrases)

    def build_any_grid(self, idx_signals, n_chars):
            master_grid = []
            for i, phrase in enumerate(idx_signals):
                this_grid = torch.zeros(len(idx_signals[0]), n_chars)
                for w, word in enumerate(phrase):
                    this_grid[w][word] = 1
                master_grid.append(this_grid)
            return torch.stack(master_grid)

    def tensors_from_preproc(self, enc_lang, dec_lang, examples):
        self.xs, self.ys = [], []
        for example in examples:
            src_tensor = self.make_hot(
                self.tensorFromSentence(enc_lang, example.source),
                enc_lang.n_words
            ).view(-1)

            tgt_tensor = self.make_hot(
                self.tensorFromSentence(dec_lang, example.target),
                dec_lang.n_words
            ).view(-1)
            example.source_tensor = src_tensor
            example.target_tensor = tgt_tensor
            self.xs.append(src_tensor)
            self.ys.append(tgt_tensor)
    
        try:
            self.create_batched_tensors(self.xs, self.ys)

        except Exception as e:
            print(f"!Examples not all the same length, stacked tensors not created!")

    def create_batched_tensors(self, xs, ys):
        tensor_xs = torch.stack(xs)
        tensor_ys = torch.stack(ys)
        self.tensors = [tensor_xs, tensor_ys]
        self.input_tensors, self.output_tensors = tensor_xs, tensor_ys
        

    def __len__(self):
        if self.data_scaling:
            scaled_len = len(self.xs) * self.data_scaling
        else:
            scaled_len = len(self.xs)

        return scaled_len

    def __getitem__(self, k):
        k = k % len(self.examples)
        return tuple(
            [self.examples[k].source_tensor,
            self.examples[k].target_tensor]
        )

        



