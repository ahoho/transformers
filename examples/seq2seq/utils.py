import itertools
import json
import linecache
import math
import os
import pickle
import socket
import random
import re
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from unidecode import unidecode

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from sentence_splitter import add_newline_to_end_of_each_sentence
from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer
from transformers.file_utils import cached_property
from transformers.modeling_bart import shift_tokens_right
from penman import layout, Graph
from penman.model import Model
from penman.codec import PENMANCodec

from transformers import BartTokenizer, T5Tokenizer

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def create_shuffled_data(data_dir, output_dir, seed, shuffle_eval=False):
    """
    Create a directory of shuffled data.
    Automatically created adjacent to the original source
    """
    data_dir = Path(data_dir)
    shuffled_data_dir = Path(output_dir, str(seed))
    if not shuffled_data_dir.exists():
        shuffled_data_dir.mkdir()
        for split in ['train', 'val', 'test']:
            src = (data_dir / f"{split}.source").read_text()
            tgt = (data_dir / f"{split}.target").read_text()
            if split == "train" or shuffle_eval:
                random.seed(seed)
                paired_train = list(zip(src.split("\n"), tgt.split("\n")))
                random.shuffle(paired_train)
                src, tgt = zip(*paired_train)
                src, tgt = "\n".join(src), "\n".join(tgt)
            (shuffled_data_dir / f"{split}.source").write_text(src)
            (shuffled_data_dir / f"{split}.target").write_text(tgt)
    return shuffled_data_dir


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    if not isinstance(refs_lns[0], list):
        refs_lns = [refs_lns]
    output_lns_normed = [
        ' '.join(re.split('(\W)', unidecode(line.lower())))
        for line in output_lns
    ]
    refs_lns_normed  = [
        [
            ' '.join(re.split('(\W)', unidecode(line.lower())))
            for line in ref
        ] for ref in refs_lns
    ]
    return {
        "bleu": corpus_bleu(output_lns, refs_lns, **kwargs).score,
        "bleu_normed": corpus_bleu(output_lns_normed, refs_lns_normed, **kwargs).score
    }


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        rouge: Dict = calculate_rouge(pred_str, label_str)
        summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        rouge.update({"gen_len": summ_len})
        return rouge

    def translation_metrics(pred: EvalPrediction) -> Dict:
        pred_str, label_str = decode_pred(pred)
        bleu: Dict = calculate_bleu(pred_str, label_str)
        gen_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        bleu.update({"gen_len": gen_len})
        return bleu

    compute_metrics_fn = summarization_metrics if "summarization" in task_name else translation_metrics
    return compute_metrics_fn

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.src_lens[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.src_lens[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def encode_line(self, tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
        """Only used by LegacyDataset"""
        return tokenizer(
            [line],
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
        )

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["decoder_input_ids"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
        }
        return batch

class PenmanDataset(LegacySeq2SeqDataset):
    """
    Handle AMR graphs in Penman, allowing for shuffling of representation
    """
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        graph_shuffling=None,
        graph_reordering=False,
        shuffle_during_gen=True,
        shuffle_eval=False,
        shuffle_consistently=False,
        append_second_graph=None,
        graph_masking=None,
        graph_masking_mixture=0.5,
        graph_token_masking_prob=0.2,
        surface_in_masked_input=False,
        batch_by_task=False,
        eval_seed=42,
    ):
        super().__init__(
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            type_path=type_path,
            n_obs=n_obs,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            prefix=prefix,
        )
        self.type_path = type_path
        self.eval_seed = eval_seed
        self.amr_codec = PENMANCodec()
        self.sense_pattern = re.compile('-[0-9][0-9]$')
        self.shuffle_during_gen = shuffle_during_gen
        self.shuffle_consistently = shuffle_consistently
        self.append_second_graph = append_second_graph
        self.edge_types = set(
            t for t in Path(self.src_file).read_text().split() if t.startswith(":")
        )

        # graph masking
        self.graph_masking_mixture = graph_masking_mixture
        self.graph_token_masking_prob = graph_token_masking_prob
        self.surface_in_masked_input = surface_in_masked_input
        self.batch_by_task = batch_by_task
        self.do_graph_completion_batch = False # start false

        if type_path == "train" or shuffle_eval:
            self.graph_shuffling = graph_shuffling
            self.graph_masking = graph_masking
            self.graph_reordering = graph_reordering
        elif graph_masking_mixture == 1:
            self.graph_shuffling = None
            self.graph_masking = graph_masking
            self.graph_reordering = None
        else:
            self.graph_shuffling = None
            self.append_second_graph = "canonical" if self.append_second_graph is not None else None
            self.graph_masking = None
            self.graph_reordering = None

    def randomize_graph(self, graph_repr, graph_shuffling=None):
        """
        Randomize the graph while maintaining PENMAN notation
        """
        rng = None if self.type_path == "train" and not self.shuffle_consistently else self.eval_seed
        random.seed(rng)
        
        if graph_shuffling == "rearrange":
            tree = self.amr_codec.parse(graph_repr)
            layout.rearrange(tree, key=Model().random_order) # inplace operation
            new_repr = self.amr_codec.format(tree)
        if graph_shuffling == "reconfigure":
            graph = self.amr_codec.decode(graph_repr)
            tree = layout.reconfigure(graph, key=Model().random_order)
            new_repr = self.amr_codec.format(tree)
        if graph_shuffling == "randomize":
            graph = self.amr_codec.decode(graph_repr)
            graph_random = Graph(random.sample(graph.triples, k=len(graph.triples)))
            new_repr = self.amr_codec.encode(graph_random)
    
        return new_repr

    def mask_graph(self, clean_graph, raw_graph=None, surface=None):
        """
        Mask graph components in a "text-to-text" fashion

        `self.graph_masking` behavior. 

            original input: ( want :arg0 ( boy ) :arg1 ( go :arg0 boy ) )

            # Tokens to mask (uses default T5 masking)
            "components"
            in:  ( want <X> ( boy ) :arg1  <Y> go :arg0 boy ) )
            out: <X> :arg0 <Y> ( <Z>

            "nodes"
            in:  ( <X> :arg0 ( boy ) :arg1 ( go :arg0 <Y> ) )
            out: <X> want <Y> boy <Z> 

            "all"
            in: ( <X> ( boy ) :arg1 <Y> go :arg0 boy ) )
            out: <X> want :arg0 <Y> ( <Z>

            # Other masking styles
            "mass":
            in: ( <X> ( boy ) :arg1 <X> go :arg0 boy ) )
            out: original text

            "drop":
            in: ( ( boy ) :arg1 go :arg0 boy ) )
            out: original text

            "corrupt":
            in: ( ) boy ) :arg0 go :arg1 boy ))
            out: original text

            "unshuffle":
            if input is "reconfigure" or "randomize", the output is unshuffled.
            Must be used in combination with "mass", "drop", or "corrupt"

            "surface":
            more-or-less standard MLM on the surface sentence,
            combining with other variants does nothing

        If `self.surface_in_input` is True, then the surface form of the sentence is
        also included
        """
        rng = None if self.type_path == "train" else self.eval_seed
        random.seed(rng)

        if "components" in self.graph_masking:
            components = {"(", ")"} | self.edge_types
            masked_source, target = self.mask_example(clean_graph, components)
        if "nodes" in self.graph_masking:
            nodes = set(clean_graph.split()) - ({"(", ")"} | self.edge_types)
            masked_source, target = self.mask_example(clean_graph, nodes)
        if "all" in self.graph_masking:
            masked_source, target = self.mask_example(clean_graph)
    
        if "mass" in self.graph_masking: # e.g., "components-mass"
            masked_source = re.sub("<extra_id_[0-9]+>", "<extra_id_0>", masked_source)
            target = clean_graph
        if "drop" in self.graph_masking:
            masked_source = re.sub("<extra_id_[0-9]+>", "", masked_source)
            target = re.sub("<extra_id_[0-9]+>", "", target)
        if "corrupt" in self.graph_masking: # currently only makes sense for "components"
            parens = {"(", ")", ""}
            edges = {":ARG0", ":ARG1", ":ARG2", ":op1", ":mod", ":ARG0-of", ":ARG1-of"}
            source_toks = clean_graph.split()
            for idx, tok in enumerate(source_toks):
                if random.random() < self.graph_token_masking_prob:
                    if tok in parens:
                        source_toks[idx] = random.choice(list(parens - {tok}))
                    if tok.startswith(":"):
                        source_toks[idx] = random.choice(list(edges - {tok}))
            masked_source = " ".join(source_toks)
            target = clean_graph
        
        if "unshuffle" in self.graph_masking: # e.g., "components-corrupt-unshuffle"
            target = self.simplify_graph(raw_graph)

        if "surface" in self.graph_masking: # e.g., "surface"
            masked_source, target = self.mask_example(surface)

        if self.surface_in_masked_input:
            masked_source = f"{surface} <GRAPH> {masked_source}"

        return masked_source, target


    def reorder_graph(self, clean_graph, raw_graph, possibly_shuffled_graph, surface):
        """
        Reorder a graph according to various schemes.
        Args:
            clean_graph: cleaned version of possibly_shuffled_graph
            raw_graph: raw input from file
            possibly_shuffled_graph: a shuffled version of the raw inupt, if shuffling
            surface: target sentence
        """
        if self.graph_reordering == "reorder":
            source = clean_graph
            target = self.simplify_graph(raw_graph)
        if self.graph_reordering == "generate":
            target = f"{surface} <GRAPH> {self.simplify_graph(raw_graph)}"

        return source, target

    def mask_example(self, text, maskable_tokens=None):
        """
        Create a masked example from input text.

        Confirmed to replicate `t5.data.preprocessorsnoise_span_to_unique_sentinel`
        """
        source_toks, target_toks = [], []
        s_id, t_id = 0, 0
        inside_masking_span = True

        for tok in text.split():
            maskable = tok in maskable_tokens if maskable_tokens is not None else True
            if random.random() < self.graph_token_masking_prob and maskable:
                if not inside_masking_span or len(target_toks) == 0:
                    source_toks.append(f"<extra_id_{s_id}>")
                    s_id += 1
                target_toks.append(tok)
                inside_masking_span = True
            else:
                if inside_masking_span:
                    target_toks.append(f"<extra_id_{t_id}>")
                    t_id += 1
                source_toks.append(tok)
                inside_masking_span = False

        return " ".join(source_toks), " ".join(target_toks)

    def simplify_graph(self, graph_repr):
        """
        Borrowed from dualgraph's preproc_amr.py, removes extraneous info from graph

        github.com/UKPLab/emnlp2019-dualgraph
        """
        graph = self.amr_codec.decode(graph_repr)
        instance_map = {
            i.source: i.target for i in graph.instances() if i.target is not None
        }
        tokens = graph_repr.split()

        new_tokens = []

        for prev_tok, tok in zip([None] + tokens[:-1], tokens):
            # ignore wiki
            if prev_tok == ":wiki" or tok == ":wiki":
                continue

            # ignore instance-of
            if tok.startswith('('):
                new_tokens.append('(')
                continue
            elif tok == '/':
                instance_declaration = True
                continue
            
            # predicates, we remove any alignment information and parenthesis
            elif tok.startswith(':'):

                new_tok = tok.strip(')')
                new_tokens.append(new_tok)

                count_ = tok.count(')')
                for _ in range(count_):
                    new_tokens.append(')')

            # concepts/reentrancies, treated similar as above
            else:
                new_tok = tok.strip(')')
                new_tok = new_tok.split('~')[0]
                # now we check if it is a concept or a variable (reentrancy)
                # need to check for "instance-of" because of first person pronoun "I"
                if new_tok in instance_map and not prev_tok == '/':
                    # reentrancy: replace with concept
                    new_tok = instance_map[new_tok]

                # remove sense information
                elif re.search(self.sense_pattern, new_tok):
                    new_tok = new_tok[:-3]
                # remove quotes
                elif new_tok[0] == '"' and new_tok[-1] == '"':
                    new_tok = new_tok[1:-1]
                new_tokens.append(new_tok)

                count_ = tok.count(')')
                for _ in range(count_):
                    new_tokens.append(')')

        return ' '.join(new_tokens)

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch = super().collate_fn(batch)
        batch["complete_graph"] = self.do_graph_completion_batch
        self.do_graph_completion_batch = random.random() < self.graph_masking_mixture

        return batch

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1

        raw_graph_repr = linecache.getline(str(self.src_file), index).rstrip("\n")

        target_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert raw_graph_repr, f"empty source line for index {index}"
        assert target_line, f"empty target line for index {index}"
        prefix = self.prefix

        # use a graph-completion (masking or reorder) objective
        do_graph_completion = random.random() < self.graph_masking_mixture
        if self.batch_by_task:
            do_graph_completion = self.do_graph_completion_batch # set in `collate_fn()`

        # if both masking and reordering on, do either one or the other at random
        do_mask = self.graph_masking is not None
        if self.graph_masking and self.graph_reordering:
            do_mask = random.random() < 0.5

        # randomize the graph
        possibly_shuffled_graph = raw_graph_repr
        if self.graph_shuffling is not None and ((random.random() < self.shuffle_during_gen) or do_graph_completion):
            prefix = f"{self.graph_shuffling} Graph:"
            possibly_shuffled_graph = self.randomize_graph(raw_graph_repr, self.graph_shuffling) 
        clean_graph_repr = self.simplify_graph(possibly_shuffled_graph)
            
        # append a second representation, if desired
        if self.append_second_graph is not None:
            second_graph_repr = raw_graph_repr
            if self.append_second_graph != "canonical":
                second_graph_repr = self.randomize_graph(
                    raw_graph_repr, self.append_second_graph
                )
            clean_second_graph_repr = self.simplify_graph(second_graph_repr)
            graph_a, graph_b = clean_graph_repr, clean_second_graph_repr
            if self.type_path == "train" and random.random() > 0.5:
                graph_a, graph_b = graph_b, graph_a
            clean_graph_repr = f"{graph_a} <GRAPH> {graph_b}"
        
        # include a masking objective
        if self.graph_masking and do_graph_completion and do_mask:
            prefix = f"{self.graph_masking} Graph: "
            clean_graph_repr, target_line = self.mask_graph(
                clean_graph_repr, raw_graph_repr, surface=target_line
            )

        # reorder a shuffled input
        if self.graph_reordering and do_graph_completion and not do_mask:
            prefix = f"{self.graph_reordering} Graph: " if self.graph_masking_mixture <= 1 else prefix
            clean_graph_repr, target_line = self.reorder_graph(
                clean_graph_repr, raw_graph_repr, possibly_shuffled_graph, target_line,
            )

        source_line = prefix + clean_graph_repr
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, target_line, self.max_target_length)
        # TODO: drop if too long?
        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }


class WebNLGShuffleDataset(LegacySeq2SeqDataset):
    """
    Shuffle linearized knowledge graphs (KG) by triple and, within each triple, by entity
    """
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        shuffle_components=True,
        component_break="<entity>",
        shuffle_spo=False,
        spo_regex="<[SPO]>[^<]+",
        shuffle_eval=False,
        reconstruct_graph_prob=0.,
        mlm_example_prob=0.,
        eval_seed=42,
    ):
        super().__init__(
            tokenizer=tokenizer,
            data_dir=data_dir,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
            type_path=type_path,
            n_obs=n_obs,
            src_lang=src_lang,
            tgt_lang=src_lang,
            prefix=prefix,
        )
        self.type_path = type_path
        self.eval_seed = eval_seed
        self.reconstruct_graph_prob = reconstruct_graph_prob if type_path == "train" else 0.
        self.mlm_example_prob = mlm_example_prob if type_path == "train" else 0.

        if type_path == "train" or shuffle_eval:
            self.shuffle_components = shuffle_components
            self.shuffle_spo = shuffle_spo
            self.component_break = component_break
            self.spo_regex = spo_regex
        else:
            self.shuffle_components = False
            self.shuffle_spo = False


    def shuffle_graph_components_in_line(self, source_line):
        """
        Randomize the linearization of the graph
        """
        # TODO: this is pretty brittle, just change inputs to json or something
        rng = None if self.type_path == "train" else self.eval_seed
        random.seed(rng)

        components = re.split(self.component_break, source_line)
        components = [c for c in components if c]
        random.shuffle(components)
        if self.shuffle_spo:
            components_with_shuffled_spo = []
            for spo in components:
                spo_split = re.findall(self.spo_regex, spo)
                assert(len(spo_split) == 3)
                random.seed(rng)
                shuffled_spo = " ".join(random.sample(spo_split, 3))
                components_with_shuffled_spo.append(shuffled_spo)
            components = components_with_shuffled_spo
        source_line = f" {self.component_break} ".join([''] + components)
        return source_line

    def mask_triples(self, source_line):
        """
        Mask out a random triple
        """
        # HACK: again, _very_ brittle, specific to WebNLG/T5
        rng = None if self.type_path == "train" else self.eval_seed
        random.seed(rng)

        # get out the triples, picking one
        components = re.split(self.component_break, source_line)
        components = [c for c in components if c]
        triple_to_mask = random.choice(components)

        # split the entities in the triple
        entities = re.split("(<[SPO]>)", triple_to_mask)[1:]
        assert((len(entities) == 6) & (entities[0] == "<S>"))
        idx = random.choice([1, 3, 5])
        source_line = ' '.join(entities[:idx] + ['<extra_id_0>'] + entities[idx+1:])
        target_line = ' '.join(['<extra_id_0>', entities[idx], '<extra_id_1>' if idx < 5 else ''])

        return source_line, target_line

    def mask_target(self, source_line, tgt_line):
        """
        Mask out the node entities in the target sentence
        """
        # HACK: again, _very_ brittle, specific to WebNLG/T5
        rng = None if self.type_path == "train" else self.eval_seed
        random.seed(rng)

        # get out all triples
        components = re.split(self.component_break, source_line)
        i = 0
        tgt_line_context = tgt_line
        tgt_line = ""

        for triple in components:
            if triple:
                for entity in re.split("<[SPO]>", triple)[1:]:
                    entity = entity.strip()
                    if re.search(entity, tgt_line_context, flags=re.IGNORECASE):
                        tgt_line_context = re.sub(
                            entity, f"<extra_id_{i}>", tgt_line_context, flags=re.IGNORECASE
                        )
                        tgt_line = f"{tgt_line} <extra_id_{i}> {entity}"
                        i += 1
        
        tgt_line = f"{tgt_line} <extra_id_{i}>"
        source_line = f"{source_line} <relation> {tgt_line_context} </relation>"
        return source_line, tgt_line

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        prefix = "translate Graph to Text: "
        if self.shuffle_components:
            source_line = self.shuffle_graph_components_in_line(source_line)
        if random.random() < self.reconstruct_graph_prob:
            prefix = "reconstruct Graph: "
            source_line, tgt_line = self.mask_triples(source_line)
        if random.random() < self.mlm_example_prob:
            prefix = "complete Text with Graph: "
            source_line, tgt_line = self.mask_target(source_line, tgt_line)

        source_line = prefix + source_line
        source_inputs = self.encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = self.encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, tpu_num_cores=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        if hasattr(self.tokenizer, "prepare_seq2seq_batch"):
            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )
        else:
            input_ids = torch.stack([x["input_ids"] for x in batch])
            attention_mask = torch.stack([x["attention_mask"] for x in batch])
            labels = torch.stack([x["labels"] for x in batch])

            labels = trim_batch(labels, self.pad_token_id)
            input_ids, attention_mask = trim_batch(input_ids, self.pad_token_id, attention_mask=attention_mask)

        if isinstance(self.tokenizer, T5Tokenizer):
            decoder_input_ids = self._shift_right_t5(labels)
        else:
            decoder_input_ids = shift_tokens_right(labels, self.pad_token_id)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
        }
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding="max_length" if self.tpu_num_cores is not None else "longest",  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class EndSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, size, end):
        self.size = size
        self.end = end

    def __iter__(self):
        return iter(range(self.end - self.size, self.end))

    def __len__(self):
        return self.size


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    try:
        repo = git.Repo(search_parent_directories=True)
        repo_infos = {
            "repo_id": str(repo),
            "repo_sha": str(repo.head.object.hexsha),
            "repo_branch": str(repo.active_branch),
            "hostname": str(socket.gethostname()),
        }
        return repo_infos
    except TypeError:
        return {
            "repo_id": None,
            "repo_sha": None,
            "repo_branch": None,
            "hostname": None,
        }


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
    pred_lns: List[str],
    tgt_lns: List[str],
    use_stemmer=True,
    rouge_keys=ROUGE_KEYS,
    return_precision_and_recall=False,
    bootstrap_aggregation=True,
    newline_sep=True,
) -> Dict:
    """Calculate rouge using rouge_scorer package.

    Args:
        pred_lns: list of summaries generated by model
        tgt_lns: list of groundtruth summaries (e.g. contents of val.target)
        use_stemmer:  Bool indicating whether Porter stemmer should be used to
        strip word suffixes to improve matching.
        rouge_keys:  which metrics to compute, defaults to rouge1, rouge2, rougeL, rougeLsum
        return_precision_and_recall: (False) whether to also return precision and recall.
        bootstrap_aggregation: whether to do the typical bootstrap resampling of scores. Defaults to True, if False
            this function returns a collections.defaultdict[metric: list of values for each observation for each subscore]``
        newline_sep:(default=True) whether to add newline between sentences. This is essential for calculation rougeL
        on multi sentence summaries (CNN/DM dataset).

    Returns:
         Dict[score: value] if aggregate else defaultdict(list) keyed by rouge_keys

    """
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    else:
        return aggregator._scores  # here we return defaultdict(list)


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
