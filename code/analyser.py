import torch
import torch.nn as nn
import numpy as np

from itertools import combinations, product
from Levenshtein import distance
from math import ceil
from scipy import stats
from torch.multiprocessing import Pool
from collections import defaultdict
from random import choices, sample
from torch import tensor, Tensor
from copy import deepcopy as dc
from string import ascii_lowercase, punctuation, digits, ascii_uppercase
from operator import index
from tkinter import N

from utils import prior_work


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Analyser(object):
    
    def __init__(self, config, game, data):
        self.config = config
        self.game = game

        self.data = data

        self.n_roles = config.n_roles
        self.n_atoms = config.n_atoms
        self.signals = None
        self.logits = None
        self.semantics = None
        self.reconstructions = None
        self.idx_reconstructions =  None
        self.alpha_semantics = None
        self.alpha_signals = None


    def get_interactions(self, dataset, get_hiddens=True, on_cpu=False):
        ''''
        input: dataset, expected as a single stack of tensors

        Generates language for a given dataset collecting interactions
        for each input. Separates into signals, reconstructions, and logits

        todo: make work for non-stacked (batched)
        
        '''
        
        by_attribute_input = dataset.view(len(dataset), self.n_roles, self.n_atoms)
        labels = by_attribute_input.argmax(dim=-1).view(len(dataset) * self.n_roles)

        if not on_cpu:
            dataset = dataset.to(self.config.device)
            self.game = self.game.to(self.config.device)
            labels = labels.to(self.config.device)

        self.game.sender.training = False
        self.game.receiver.training = False
        with torch.no_grad():
            self.loss, self.interaction = self.game(dataset, labels)

            self.signals = self.interaction.message
            self.reconstructions = self.interaction.receiver_output
            self.idx_reconstructions = self.data.logits_to_idx(self.interaction.receiver_output)
            self.idx_semantics = self.data.logits_to_idx(dataset)

            self.game.sender.return_raw = True
            signals, self.logits, entropys = self.game.sender(dataset)
            self.game.sender.return_raw = False
            if get_hiddens:
                self.game.sender.return_encoding = True
                self.sender_hidden = self.game.sender(dataset)
                self.game.sender.return_encoding = False
                self.game.receiver.return_encoding = True
                self.receiver_hidden = self.game.receiver(self.signals)
                self.game.receiver.return_encoding = False


        self.game.sender.training = True
        self.game.receiver.training = True

    #Topographic Similarity
    #Performs topsim along with a number of helper functions

    def idx_to_string(self, array: list):
        '''
        Converts list of indexes to alphanumeric string
        for use with LevDistance

        array: list of list of idx semantics [[2,6,9]]

        return: list of stings
        '''
        characters = ascii_lowercase+punctuation+digits+ascii_uppercase
        
        all_strings = []
        for idxes in array:
            a_string = []
            for idx in idxes:
                
                a_string.append(characters[int(idx)])
            all_strings.append(''.join(a_string))

        return all_strings

    def permute_distances(self, array: list):
        '''
        Generates all possible pairings of examples and calculates their distances
        
        array: list of alpha numeric strings

        returns: list of distances 2**n long
        
        '''
        pairs = combinations(array, 2)
        distances = []
        for pair in pairs:
            dist = distance(pair[0], pair[1])
            if not (type(dist) == int):
                print(f'{pair} failed with {dist}')
            distances.append(dist)

        return distances

    def top_sim(self, signals: list, semantics, subsample=0.25, spear=True):
        '''
        Calculates topographic similarity of a language

        signals: list of list of idx signals [[2,3,6]]
        semantics:  list of list of idx semantics [[1,2,3]]
        spear: if true spearman correlation else pearson
        parallel: if true spreads job across cpu cores

        returns: topsim (float)
        '''

        if type(semantics) == torch.Tensor:
            semantics = self.data.logits_to_idx(semantics).tolist()

        if subsample:
            num_examples = int(len(signals)*subsample)
            sample_ids = sample(list(range(len(signals))), num_examples)
    
            signals = [signals[i] for i in sample_ids]
            semantics = [semantics[i] for i in sample_ids]

        self.alpha_signals = self.idx_to_string(signals)
        self.alpha_semantics = self.idx_to_string(semantics)

        sig_dist = self.permute_distances(self.alpha_signals)
        sem_dist = self.permute_distances(self.alpha_semantics)

        if spear:
            self.topsim = stats.spearmanr(sig_dist, sem_dist)[0]
        else:
            self.topsim = stats.pearsonr(sig_dist, sem_dist)[0]

        return self.topsim



    def z_scoring(
        self, sem_dist: torch.tensor, rep_sim: torch.tensor, 
        r_score: float, repeat_z=50
    ):
        '''
        Z-scores to support topsim calculations, randomly permutes distances 
        then performs a correlation test to see what the defaul correlation is.
        Returns number of standard deviations from permuted score to true score

        sem_dist: (tensor) distance between semantics
        rep_sim: (tensor) distance between representations
        r_score: (float) topsim valie
        repeat_z: (int) number of times to permute distances
        '''
        all_z, all_zp = [], []
        for i in range(repeat_z):
            idx = torch.randperm(rep_sim.shape[0])
            rep_sim = rep_sim[idx].view(rep_sim.size())
            z_value, zp_value = stats.spearmanr(rep_sim.detach(), sem_dist)
            all_z.append(z_value)
            all_zp.append(zp_value)

        z_std = torch.tensor(all_z).std().item()
        zp_mean = torch.tensor(all_zp).mean().item()
        z_dist = abs(r_score)/z_std
        return z_dist, zp_mean



    #Positional Confidence Scoring

    def mle_probabilities(self, dataset, signals):
        '''
        Estimates the raw counts for how often roles, atoms and characters
        each occur, and occur together. 
        '''
        semantic_roles = defaultdict(lambda: defaultdict(lambda: 0.0))
        characters = defaultdict(lambda: 0.0)
        atoms = defaultdict(lambda: 0.0)

        letters_given_atoms = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: 0.0
                        )
                    )
                )
            )
        letter_by_position = defaultdict(lambda: defaultdict(lambda: 0))

        for example_id, example in enumerate(dataset.examples):
            for semantic_role, atom in zip(example.dep_tags, example.source):
                for signal_position, letter in enumerate(signals[example_id][:-1]): #slice omits eos token

                    letters_given_atoms[semantic_role][atom][signal_position][letter] \
                        = letters_given_atoms[semantic_role][atom][signal_position][letter] + 1
                    
                    letter_by_position[signal_position][letter] \
                        = letter_by_position[signal_position][letter] +1

                    characters[letter] = characters[letter] + 1
                    
                semantic_roles[semantic_role][atom] = semantic_roles[semantic_role][atom] + 1
                atoms[atom] = atoms[atom]+1
        

        return semantic_roles, characters, atoms, letters_given_atoms, letter_by_position

    def conditional_probabilities(
        self, 
        semantic_roles: defaultdict, 
        characters: defaultdict, 
        atoms: defaultdict, 
        letters_given_atoms: defaultdict, 
        letter_by_position: defaultdict
        ):
        '''
        Uses raw counts to generate conditional probabilities for 
        char | position, atom, role & atom | char, position, role
        
        '''

        letter_given_atom_co_prob = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: 0.1
                        )
                    )
                )
            )
        atom_given_letter_co_prob = defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: 0.1
                        )
                    )
                )
            )

        for semantic_role in letters_given_atoms:
            for atom in letters_given_atoms[semantic_role]:
                for signal_position in letters_given_atoms[semantic_role][atom]:
                    for character in letters_given_atoms[semantic_role][atom][signal_position]:

                        letter_given_atom_co_prob[semantic_role][atom][signal_position][character] \
                            = letters_given_atoms[semantic_role][atom][signal_position][character] \
                                /semantic_roles[semantic_role][atom]

                        atom_given_letter_co_prob[semantic_role][signal_position][character][atom] \
                            = letters_given_atoms[semantic_role][atom][signal_position][character] \
                                /letter_by_position[signal_position][character]
            
        return letter_given_atom_co_prob, atom_given_letter_co_prob

    def holistic_signals(self, dataset, n_chars=26, signal_len =6):
        '''
        Generates a random unique signal for each meaning, given a signal length
        and character space. Useful for comparisons with maximal variation
        '''
        signal_space = list(product(range(n_chars), repeat=signal_len))
        tuple_signals = sample(signal_space, k=len(dataset.examples)) 
        random_signals = []
        for i in tuple_signals:
            random_signals.append(list(i))
            random_signals[-1].append(0)

        return random_signals


    def homonymy(self, prob_table: list):
        '''
        
        !uses role isolation!
        Takes a list of prob Tensors each n_atoms X N_signal_len X 
        n_chars. Refactors to len X chars X atoms, concatenates all roles.
        Normalizes then assesses entropy on the last axis, and means across all 
        positions to get language level homonomy.

        Note: This drops all unused characters before normalizing

        prob_table: list of tensors, len n_roles
            dimensions n_atoms X n_signal_len X n_chars
        '''
    
        all_role_postions = []
        for role in prob_table:
            letters_last = role.permute(1,2,0)
            all_letters = torch.flatten(letters_last, start_dim=-1)
            by_position = [position for position in all_letters]
            all_role_postions.append(by_position)

        lang_homonomy = []
        for role in all_role_postions:
            role_homonomy = []
            max_entropies = []
            for position in role:
                non_zero_positions = position[position.sum(-1)>0]
                if non_zero_positions.shape[0] == 0:
                    position = torch.ones_like(position)
                    non_zero_positions = position 
            
                max_entropy = self.max_entropy(non_zero_positions)
                max_entropies.append(max_entropy)

                entropy = torch.distributions.Categorical(
                    non_zero_positions
                ).entropy()

                entropy = (entropy.mean())/max_entropy
                role_homonomy.append(entropy)

            lang_homonomy.append(min(role_homonomy))
            

        return float(np.mean(lang_homonomy))


    def synonymy(self, prob_table: list):
        '''
        Takes a list of probability tensors from a normalized PMI tensor, 
        calculates entropy on the last dimension. Divides the max by the entropy
        of a uniform distribution and means across all values to get the 
        language level synonymy

        prob_table: list of tensors, len n_roles
            dimensions n_atoms X n_signal_len X n_chars
        '''
        role_entropies = []
        for semantic_role in prob_table:
            atom_entropies = []
            for atom in semantic_role:
                non_zero_positions = atom[atom.sum(-1)>0]
                if len(non_zero_positions) == 0:
                    non_zero_positions = torch.ones_like(atom)
                entropy = torch.distributions.Categorical(non_zero_positions).entropy()
                max_entropy = self.max_entropy(atom)
                
                atom_synonymy = entropy.min()/max_entropy
                atom_entropies.append(atom_synonymy)

            role_entropies.append(float(np.mean(atom_entropies)))
        
        language_synonymy = float(np.mean(role_entropies))

        return language_synonymy
    
    def freedom(self, prob_table: list):
        '''
        Assess atom order freedom for a given probability table

        '''
        role_uniformity = []

        for semantic_role in prob_table:
            entropy = torch.distributions.Categorical(semantic_role).entropy()
            entropy = entropy.mean(0) #collapse across all atoms in each position
            max_entropy = self.max_entropy(semantic_role)
            divergent_entropy = entropy.min(dim=-1).values
            bounded_divergent_entropy = divergent_entropy/max_entropy

            role_uniformity.append(bounded_divergent_entropy.item())

        language_uniformity = float(np.mean(role_uniformity))

        return language_uniformity

    def entanglement(self, prob_table: list):
        mean_entropies, maxes = [], []
        max_entropy = 0
        for semantic_role in prob_table:
            max_entropy = self.max_entropy(semantic_role)
            try:
                entropy = torch.distributions.Categorical(semantic_role).entropy()
            except:
                print("entaglement entropy failed, using norm or uniform")
                print("means some n_grams unattested - measure unreliable")
                print("(try a larger sample of data)")
                norm_role = self.norm_or_uniform(semantic_role)
                entropy = torch.distributions.Categorical(norm_role).entropy()
            mean_entropy = entropy.mean(dim=0)
            mean_entropies.append(mean_entropy)
            maxes.append(
                semantic_role.max(-1).values.mean(0)
            )
            
        pairs = combinations(mean_entropies, r=2)
        
        differences = []
        for pair in pairs:
            diff = abs(pair[0] - pair[1])
            max_index = diff.argmax()
            pair_max = (max([pair[0].max(),pair[1].max()])).item()
            pair_max = pair_max if pair_max>0 else 0.000001
            differences.append(float(1-(diff.max().item()/pair_max)))

        pairs = combinations(maxes, r=2)
        max_differences = []    
        for pair in pairs:
            diff = abs(pair[0] - pair[1])
            max_index = diff.argmax()
            pair_max = (max([pair[0].max(),pair[1].max()])).item()
            pair_max = pair_max if pair_max>0 else 0.000001
            max_differences.append(float(1-(diff.max().item()/pair_max)))
            
        entanglement = float(np.mean(differences))
        entanglement_max = float(np.mean(max_differences))
   
        return entanglement

    def full_prob_analysis(
        self, 
        dataset, 
        signals, 
        n_gram_size : int = 1,
        split_name: str = "train",
    ):
        '''
        Runs the full suite of our 4 variation measures, with residual entropy
        and returns a dictionary of results
        '''
        results = {}

        mean_pcs, prob_dict, co_prob_tensor, word_order \
            = self.create_prob_table(dataset, signals)

        results[f'{split_name}_word_order'] = word_order

        results[f'{split_name}_entanglement'] = self.entanglement(co_prob_tensor)
        results[f'{split_name}_freedom'] = self.freedom(co_prob_tensor)
        results[f'{split_name}_homonomy'] = self.homonymy(co_prob_tensor)
        results[f'{split_name}_synonymy'] = self.synonymy(co_prob_tensor)
        

        residual_entropy = self.residual_entropy(self.idx_semantics, co_prob_tensor, self.signals)
        results[f'{split_name}_residual_entropy'] = residual_entropy

        return results


    def create_prob_table(self, dataset, signals):
        ''''
        Estimates the raw counts for how often roles, atoms and characters
        each occur, and occur together. Then uses these to create a full probability
        table with dimensions roles x atoms x positions x chars
        '''
        #calc probs
        semantic_roles, characters, atoms, letters_given_atoms, letter_by_position \
            = self.mle_probabilities(dataset, signals)        
            
        #calc co_probs
        letter_given_atom_co_prob, atom_given_letter_co_prob \
            = self.conditional_probabilities(
                semantic_roles,
                characters,
                atoms,
                letters_given_atoms,
                letter_by_position
            )

        self.letter_given_atom_co_prob = letter_given_atom_co_prob

        co_prob_tensors = self.convert_dict_to_tensors(dataset, letter_given_atom_co_prob)
        self.mean_pcs = self.mean_co_probs_by_position(co_prob_tensors)
        word_order = self.estimate_word_order(self.mean_pcs, letter_given_atom_co_prob)

        return self.mean_pcs, letter_given_atom_co_prob, co_prob_tensors, word_order
    
    def convert_dict_to_tensors(self, dataset, co_probs: defaultdict, n_size: int = 1):
        '''
        converts the probability table, comprised of nested dicts to a list of
        tensors 1 per role. Note that it's a LIST of tensors to straight-forwardly 
        allow each role to have a different number of atoms (i.e. 10 subjects and 20 verbs)
        '''
        dict_dimensions = self.fetch_dims(dataset, n_size=n_size)
        tensors = self.create_tensors_from_dims(dict_dimensions)
        

        for sem_ix, semantic_role in enumerate(co_probs):
            for atom_ix, atom in enumerate(co_probs[semantic_role]):
                for pos_ix, signal_position in enumerate(co_probs[semantic_role][atom]):
                    for char_ix, character in enumerate(co_probs[semantic_role][atom][signal_position]):
                        tensors[sem_ix][atom_ix][pos_ix][int(character)] \
                            = co_probs[semantic_role][atom][signal_position][character]

        return [tens.float() for tens in tensors]

    def atomid2tensoridx(self, dataset, co_probs):
        '''
        Returns a pair of dictionaries for mapping between a atom and the index
        of the tensor where the corresponding probability distribution can be found
        '''
        atomid2tensoridx = defaultdict(lambda: defaultdict(lambda: 0))
        tensoridx2atom = defaultdict(lambda: defaultdict(lambda: 0))

        for sem_ix, semantic_role in enumerate(co_probs):
            for atom_ix, atom in enumerate(co_probs[semantic_role]):
                atom_id = dataset.enc_lang.atom2index[atom]
                atomid2tensoridx[sem_ix][atom_id] = atom_ix
                tensoridx2atom[sem_ix][atom_ix] = atom

        return atomid2tensoridx, tensoridx2atom
        
    def fetch_dims(self, dataset, n_size=1, config=None):
        '''
        Gets the expected dimensions of the full probability table, by checking
        the number of roles, atoms, and chars - it can handle n_grams larger than
        1 for determining the number of chars via the n_size param
        '''
        config = config if config else self.config
        try:
            n_roles = config.n_roles
            n_atoms = config.n_atoms_per_role
        except:
            print("Number of Atoms and Roles not specified, Inferring")
            n_roles = len(dataset.enc_lang.dep_words)
            n_atoms = [len(dataset.enc_lang.dep_words[role]) for role in dataset.enc_lang.dep_words]
        
        n_positions = ceil(self.config.signal_len/n_size)
        
        #size of final dim depends on n_gram size
        n_characters = 0
        combinations = product(range(self.config.signal_alphabet_size), repeat=n_size)
        for combo in combinations:
            n_characters += 1

        dims = [(n_atoms[i], n_positions, n_characters) for i in range(n_roles)]

        return dims

    def create_tensors_from_dims(self, dims: list, floor: float = 0.0):
        '''
        Dims: list of tensor dimensions, default zeros
        floor: lowest value in tensor

        used to convert nested dictionaries of known depth (dims) to a list of 
        tensors. in practice used here to go from co-prob table to a lost of prob 
        tensors with one for each role
        '''
        tensors = []
        for i in dims:
            zero_tensors = torch.zeros(i)
            tensors.append(zero_tensors+floor) #floor is added to account for unused atoms
        return tensors


    def mean_co_probs_by_position(self, co_probs: tensor, top_k=1):
        '''
        Co probs: full probability tensor
        top_k: how many probabilities per (position, atom, role) combo should be 
            meaned
        
        '''
        means_by_position = []
        for semantic_role in co_probs:
            top_correlates = torch.topk(semantic_role, top_k, dim=-1)
            top_values = top_correlates.values.squeeze()
            means_by_position.append(top_values.permute(1, 0).mean(dim=-1))
            
        return means_by_position

    def estimate_word_order(self, mean_probs: list, co_probs: defaultdict):
        '''
        mean probs: list of role probabilities per signal position
        co_probs: full probability table

        returns a list of the highest probability role in each position as an 
        estimate of atom order
        '''
        idx_word_order = torch.stack(mean_probs).permute(1,0).argmax(dim=-1).tolist()
        roles = [role for role in co_probs]

        word_order = ' '.join([roles[idx] for idx in idx_word_order])

        return word_order

    def max_entropy(self, probabilities: tensor):
        ''''
        Takes a tensor as input, and returns the entropy of a same-sized uniform
        probability distribution. Used for bounding measures between zero and one
        '''
        uniform_ones = torch.ones(probabilities.shape)
        max_entropies = torch.distributions.Categorical(uniform_ones).entropy()
        theoretical_max = max_entropies.max()
        return theoretical_max

    def min_inventory(self, prob_table, threshold=10):
        '''
        Returns the number of characters that have occur more than the threshold
        number of times for a given semantic role. Gives an idea of the percentage
        of the total signal space that's in use
        '''
        minimum = threshold/len(self.signals) if threshold else 0
        all_unused = []
        for semantic_role in prob_table:
            n_unused_chars = torch.lt(semantic_role, minimum).sum().item()
            percent_unused_chars = n_unused_chars/len(semantic_role.view(-1))
            all_unused.append(percent_unused_chars)

        return np.mean(all_unused), all_unused
    
    def norm_or_uniform(self, prob_table):
        '''
        takes a tensor of pmis and normalizes it on the last axis. If that last 
        axis is all 0s then it's replaced with a uniform distribution

        pmi_tensor: list of tensors, len n_roles
            dimensions n_atoms X n_signal_len X n_chars
        '''
        if type(prob_table) == list:
            norm_pmi = dc(prob_table)
            for semantic_role in norm_pmi:
                for atom_id, n_gram_sums in enumerate(semantic_role.sum(axis=-1).tolist()):
                    for position_id, n_gram_sum in enumerate(n_gram_sums):
                        if n_gram_sum == 0.0:
                            semantic_role[atom_id][position_id] = semantic_role[atom_id][position_id]+1
            normed_tensor = []
            for semantic_role in norm_pmi:
                normed_tensor.append(semantic_role/semantic_role.sum(-1, keepdim=True))

        else:
            for atom_id, n_gram_sums in enumerate(prob_table.sum(axis=-1).tolist()):
                    for position_id, n_gram_sum in enumerate(n_gram_sums):
                        if n_gram_sum == 0.0:
                            prob_table[atom_id][position_id] = prob_table[atom_id][position_id]+1
            
            return prob_table/prob_table.sum(-1, keepdim=True)


        return normed_tensor

    def pos_dis(self, dataset, signals):
        '''
        Positional disambiguation from Chaabouni et al. 2020
        Code copied here with minimal modification
        '''

        n_roles = self.config.n_roles
        n_atoms = self.config.n_atoms
        
        attributes = dataset.view(
            (dataset.shape[0], n_roles, n_atoms)
        ).argmax(dim=-1)

        posdis = prior_work.information_gap_representation(
            attributes,
            signals
        )

        return posdis

    def bos_dis(self, dataset, signals):
        '''
        Bag-of-symbols disambiguation from Chaabouni et al. 2020
        Code copied here with minimal modification
        '''
        n_roles = self.config.n_roles
        n_atoms = self.config.n_atoms
        vocab_size = self.config.signal_alphabet_size
        
        attributes = dataset.view(
            (dataset.shape[0], n_roles, n_atoms)
        ).argmax(dim=-1)

        histograms = prior_work.histogram(signals, vocab_size)
        bosdis =  prior_work.information_gap_representation(
            attributes, histograms[:, 1:]
        )

        return bosdis


    def residual_entropy(
        self, 
        idx_semantics, 
        int_tensor, 
        signals, 
        n_roles=3, 
        n_atoms=25, 
        remove_eos=True,
        max_substring_len=5
    ):
        '''
        This is an implementation of residual entropy as described in Resnick et al 2019
        In the setup for that paper their signal space contained only binary 
        strings - accordingly the released code for residual entropy has a number
        of design decisions which make it only suitable for binary strings. Below
        is an implementation that allows for an arbitrary-sized signal space.

        For memory concerns, it considers a maximum substring of 5 chars - this
        can be changed via the max_substring_len parameter

        '''
        
        signal_len, n_chars = self.config.signal_len, self.config.signal_alphabet_size

        meanings = idx_semantics.tolist()
        if remove_eos:
            signals = signals[:,:-1] #remove eos

        role_maxes = [role_tensor.max(-1).values for role_tensor in int_tensor]
        all_role_maxes = torch.stack(role_maxes)
        role_max_shape = all_role_maxes.shape
        flat_role_maxes = all_role_maxes.view(role_max_shape[0]*role_max_shape[1], role_max_shape[2])
        position_means = flat_role_maxes.mean(0)
        role_max_diff = flat_role_maxes - position_means


        role_position_prob = torch.zeros((n_roles, signal_len))
        for role in range(n_roles):
            role_position_prob[role] = (
                role_max_diff[role * n_atoms:(role + 1) * n_atoms]
            ).mean( axis=0)

        role_encoded_position = [[] for _ in range(n_roles)]
        role_encoded_position_values = [[] for _ in range(n_roles)]

        for position in range(signal_len):
            role = torch.argmax(role_position_prob[:, position])
            role_v = role_position_prob[:, position].max()
            role_encoded_position[int(role)].append(position)
            role_encoded_position_values[int(role)].append(float(role_v))

        for i, role in enumerate(role_encoded_position):
            if len(role)>max_substring_len:
                ind = np.argpartition(
                    role_encoded_position_values[i], 
                    -max_substring_len
                )[-max_substring_len:].tolist()
                
                role_encoded_position[i] = ind

        probs = []
        probs_n = []
        ents = np.zeros(n_roles)

        for role in range(n_roles):

            role_values = idx_semantics[:,role].unique().tolist()
            role_value_to_idx = {}
            for i, role_value in enumerate(role_values):
                role_value_to_idx[role_value] = i

            n_positions = len(role_encoded_position[role])

            if n_positions:
                signal_data = signals[:, role_encoded_position[role]].tolist()
                all_n_grams = product(range(n_chars), repeat=n_positions)
            else:
                signal_data = torch.zeros(len(signals)).unsqueeze(1).int().tolist()
                all_n_grams = [tuple([x]) for x in torch.arange(0,n_chars).tolist()]

                
            probs.append(torch.zeros((n_chars ** n_positions, n_atoms)) + 1e-8)
            probs_n.append(torch.zeros((n_chars ** n_positions, n_atoms)) + 1e-8)
            
            
            n_grams_to_idx = {}
            for n_id, n_gram in enumerate(all_n_grams):
                n_grams_to_idx[n_gram] = n_id

            for i, sub_signal in enumerate(signal_data):
                n_id = n_grams_to_idx[tuple(sub_signal)]
                value_id = role_value_to_idx[meanings[i][role]]
                probs[role][n_id, value_id] +=1

            probs_n[role] = probs[role]/probs[role].sum(1).unsqueeze(0).T
            ent = (probs_n[role]*probs_n[role].log()).sum(1)
            ents[role] = (ent*(probs[role].sum(1)/probs[role].sum())).sum()

        ent = -np.mean(ents) / np.log(25)
        return ent
