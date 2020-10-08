import itertools
import json
import linecache
import os
import pickle
import warnings
import random
import re
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import git
import numpy as np
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from unidecode import unidecode

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler

from penman import layout, Graph
from penman.model import Model
from penman.codec import PENMANCodec

from transformers import BartTokenizer, T5Tokenizer


def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt", add_eos=True):
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    line = f"{line} {tokenizer.eos_token}" if (isinstance(tokenizer, T5Tokenizer) and add_eos) else line
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
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


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class Seq2SeqDataset(Dataset):
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
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.lines_in_src_file = len(self.src_lens)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix
        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):
        return len(self.src_lens)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id) -> tuple:
        y = trim_batch(batch["decoder_input_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"])
        return source_ids, source_mask, y

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

    def make_sortish_sampler(self, batch_size):
        return SortishSampler(self.src_lens, batch_size)

    def end_of_data_sampler(self):
        return EndSequentialSampler(len(self), end=self.lines_in_src_file)


class PenmanDataset(Seq2SeqDataset):
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
        shuffle_eval=False,
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
        self.amr_codec = PENMANCodec()
        self.sense_pattern = re.compile('-[0-9][0-9]$')

        if type_path == "train" or shuffle_eval:
            self.graph_shuffling = graph_shuffling
        else:
            self.graph_shuffling = None

    def randomize_graph(self, graph_repr, graph_shuffling):
        """
        Randomize the graph while maintaining PENMAN notation
        """
        rng = None if self.type_path == "train" else self.eval_seed
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


    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        index = index + 1  # linecache starts at 1
        graph_repr = linecache.getline(str(self.src_file), index).rstrip("\n")
        orig_graph_repr = graph_repr

        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert graph_repr, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"

        # randomize the graph
        if self.graph_shuffling is not None:
            graph_repr = self.randomize_graph(orig_graph_repr, self.graph_shuffling)
        
        graph_repr = self.simplify_graph(graph_repr)

        source_line = self.prefix + graph_repr
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }


class WebNLGShuffleDataset(Seq2SeqDataset):
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
        source_eos = True
        if self.shuffle_components:
            source_line = self.shuffle_graph_components_in_line(source_line)
        if random.random() < self.reconstruct_graph_prob:
            prefix = "reconstruct Graph: "
            source_eos = False
            source_line, tgt_line = self.mask_triples(source_line)
        if random.random() < self.mlm_example_prob:
            prefix = "complete Text with Graph: "
            source_eos = False
            source_line, tgt_line = self.mask_target(source_line, tgt_line)

        source_line = prefix + source_line
        source_inputs = encode_line(
            self.tokenizer, source_line, self.max_source_length, add_eos=source_eos
        )
        target_inputs = encode_line(
            self.tokenizer, tgt_line, self.max_target_length, add_eos=True
        )

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "decoder_input_ids": target_ids,
        }


class MBartDataset(Seq2SeqDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.max_source_length != self.max_target_length:
            warnings.warn(
                f"Mbart will ignore max_target_length = {self.max_target_length} and use {self.max_source_length} for both sides."
            )

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {
            "tgt_texts": source_line,
            "src_texts": tgt_line,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_translation_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
        )
        return batch_encoding.data


class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def key(self, i):
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        idxs = np.random.permutation(len(self.data))
        sz = self.bs * 50
        ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
        sort_idx = np.concatenate([sorted(s, key=self.key, reverse=True) for s in ck_idx])
        sz = self.bs
        ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
        max_ck = np.argmax([self.key(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
        ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
        sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
        sort_idx = np.concatenate((ck_idx[0], sort_idx))
        return iter(sort_idx)

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


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}


def freeze_params(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


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
