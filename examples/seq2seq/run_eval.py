#!/usr/bin/env python

import argparse
import sys
from unidecode import unidecode
from pathlib import Path
from collections import defaultdict
import re

sys.path.append("..")

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer

from finetune import SummarizationModule, TranslationModule, DataToTextModule, ShuffledDataToTextModule, AMRToTextModule

try:
    from .utils import pickle_load, pickle_save, save_json, trim_batch, calculate_bleu, load_json
except ImportError:
    from utils import pickle_load, pickle_save, save_json, trim_batch, calculate_bleu, load_json

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_training_metrics(data):
    """
    Parse the metrics created during training
    """
    metrics = defaultdict(list)
    for step in data:
        metrics['train_val_losses'].append(step['val_avg_loss'])
        metrics['train_val_bleus'].append(step['val_avg_loss'])

    metrics['best_train_val_loss'] = float(np.min(metrics['train_val_losses']))
    metrics['best_train_val_loss_step'] = int(np.argmin(metrics['train_val_losses']))
    metrics['best_train_val_bleu'] = float(np.max(metrics['train_val_losses']))
    metrics['best_train_val_bleu_step'] = int(np.argmax(metrics['train_val_losses']))
    metrics['total_steps'] = len(data) - 1
    
    return dict(metrics)


def process_batch(batch, pad_token_id=0, device=None):
    """
    Clean batch before passing to model
    """
    if device is None:
        device = DEFAULT_DEVICE
    y = trim_batch(batch["decoder_input_ids"], pad_token_id)
    source_ids, source_mask = trim_batch(
        batch["input_ids"], pad_token_id, attention_mask=batch["attention_mask"]
    )
    return y.to(device), source_ids.to(device), source_mask.to(device)


def generate_from_model(data_loader, model, generate=True, no_pb=True):
    all_preds = []
    lls = []
    
    pad_token_id = model.tokenizer.pad_token_id

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader), disable=no_pb):
        
        y, source_ids, source_mask = process_batch(batch, pad_token_id)

        if generate:
            generated_ids = model.model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                use_cache=True,
                decoder_start_token_id=model.decoder_start_token_id,
                num_beams=5,
                min_length=0,
                max_length=model.eval_max_length,
                length_penalty=1.0,
            )
            preds = model.ids_to_clean_text(generated_ids)
            preds = [p[:p.index("<GRAPH>")] if "<GRAPH>" in p else p for p in preds]
            all_preds.extend(preds)
        if no_pb: # for beaker logs
            print(f"{i+1/len(data_loader)*100:0.2f}%")
        y = y.masked_fill(y == pad_token_id, -100)
        loss = calculate_batch_loss(model, source_ids, source_mask, y)
        lls.append(loss)
    
    lls = torch.cat(lls)
    ppl = torch.exp(lls.sum() / i).item()

    return all_preds, lls.cpu().detach().numpy(), ppl


def estimate_scaffolding_loss(model, type_path, bs, reordering, masking, shuffling, mode="bootstrap", n_samples=5):
    """
    Estimate the graph scaffolding losses
    """
    model.dataset_kwargs.update({
        'graph_masking_mixture': 1.,
        'shuffle_eval': True,
        'graph_reordering': model.hparams.amr_reordering,
        'graph_masking': masking,
        'graph_shuffling': shuffling,
    })
    pad_token_id = model.tokenizer.pad_token_id

    # OPTION A: bootstrapped estimate of sentence loss
    if mode == "bootstrap":
        sampled_lls = []
        pbar = None
        for i in range(n_samples):
            model.dataset_kwargs['eval_seed'] = i ** i
            data_loader = model.get_dataloader(
                type_path=type_path, batch_size=bs, shuffle=False
            )
            lls = []
            if pbar is None:
                pbar = tqdm(total=n_samples * len(data_loader))

            # calculate loss
            for batch in data_loader:
                y, source_ids, source_mask = process_batch(batch, pad_token_id)
                mask = y == pad_token_id
                y = y.masked_fill(mask, -100)
                loss = calculate_batch_loss(model, source_ids, source_mask, y)
                lls.append(loss)
                pbar.update()
            sampled_lls.append(torch.cat(lls))
        return torch.stack(sampled_lls).cpu().detach().numpy().T
    
    # OPTION B: mask all
    if mode == "mask_all":
        # prev_args.graph_token_masking_prob = 1.
        raise NotImplementedError("Not yet implemented")

def calculate_batch_loss(model, input_ids, attention_mask, labels):
    """
    Report cross-entropy loss for each item in the bass
    """
    mask = labels != -100
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    with torch.no_grad():
        out = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = loss_fct(out.logits.view(-1, out.logits.size(-1)), labels.view(-1))
        loss = loss.view(out.logits.size(0), -1)
        loss = loss.sum(-1) / mask.sum(-1)

    return loss


def run_generate(verbose=True):
    """

    Takes input text, generates output, and then using reference calculates the BLEU scores.

    The results are saved to a file and returned to the caller, and printed out unless ``verbose=False`` is passed.
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Directory containing model and saved args")
    parser.add_argument("output_dir", type=str, help="where to save the outputs")

    parser.add_argument("--type_path", type=str, default="val", help="Split from which to generate, i.e. <type_path>.source")
    parser.add_argument("--reference_paths", nargs="+", required=False, help="like val.target val-2.target")
    parser.add_argument("--data_dir", type=str, help="Input location (default uses that in saved args)")
    parser.add_argument("--tokenizer_path_or_name", default=None, help="Use this to match tokenization between models")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument("--shuffle_graph_components", default=False, required=False, action="store_true")
    parser.add_argument("--do_not_generate", dest="generate", default=True, action="store_false")
    parser.add_argument("--save_sentence_losses", default=False, action="store_true")
    parser.add_argument("--amr_shuffling", choices=["reconfigure", "rearrange", "randomize"], default=None)
    parser.add_argument("--append_second_amr", choices=["canonical", "reconfigure", "rearrange", "randomize"], default=None)

    parser.add_argument("--amr_masking", choices=["components", "nodes", "all"], default=None)
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )
    parser.add_argument("--output_trainining_metrics_only", default=False, action="store_true", help="Don't generate, and only up the logs from training")
    args = parser.parse_args()

    training_metrics = load_json(Path(args.model_dir, "metrics.json"))["val"]
    training_metrics = parse_training_metrics(training_metrics)
    if args.output_trainining_metrics_only:
        save_json(training_metrics, path=Path(args.output_dir, "metrics.json"))

    # Reset previously used arguments as necessary
    prev_args = pickle_load(Path(args.model_dir, "hparams.pkl"))
    prev_args.model_name_or_path = str(Path(args.model_dir, "best_tfmr"))
    prev_args.output_dir = str(args.model_dir)

    prev_args.shuffle_graph_components = args.shuffle_graph_components
    trained_with_second_amr = getattr(prev_args, "append_second_amr", None)
    prev_args.append_second_amr = args.append_second_amr
    if trained_with_second_amr and args.append_second_amr is None:
        print("Trained with a second AMR, but not set during evaluation")

    original_shuffling = prev_args.amr_shuffling
    prev_args.amr_shuffling = args.amr_shuffling

    prev_args.shuffle_graph_during_eval = (
        args.shuffle_graph_components or 
        args.amr_shuffling is not None or
        args.append_second_amr not in [None, "canonical"]
    )
    setattr(prev_args, f"n_{args.type_path}", args.n_obs)
    
    # backward-compatibility TODO: check if ok
    prev_args.amr_masking = getattr(prev_args, "amr_masking", args.amr_masking)
    prev_args.amr_reordering = getattr(prev_args, "amr_reordering", None)
    prev_args.graph_token_masking_prob = getattr(prev_args, "graph_token_masking_prob", None)
    prev_args.include_surface_in_masked_input = getattr(prev_args, "include_surface_in_masked_input", None)

    if args.data_dir is not None:
        prev_args.data_dir = args.data_dir

    if prev_args.task == "summarization":
        model = SummarizationModule(prev_args, save_hparams=False)
    elif prev_args.task == "translation":
        model = TranslationModule(prev_args, save_hparams=False)
    elif prev_args.task == "data-to-text":
        if (
            not prev_args.shuffle_graph_components
            and getattr(prev_args, "reconstruct_graph_prob", 0) == 0.
            and getattr(prev_args, "mlm_example_prob", 0) == 0.
        ):
            model: SummarizationModule = DataToTextModule(prev_args, save_hparams=False)
        else:
            prev_args.reconstruct_graph_prob = 0.
            prev_args.mlm_example_prob = 0.
            model: SummarizationModule = ShuffledDataToTextModule(prev_args, save_hparams=False)
    if prev_args.task == "amr-to-text":
        prev_args.amr_masking_mixture = 0.
        model: SummarizationModule = AMRToTextModule(prev_args, save_hparams=False)

    if args.type_path == 'test-unseen':
        model.n_obs['test-unseen'] = model.n_obs['test']
        model.target_lens['test-unseen'] = model.target_lens['test']

    data_loader = model.get_dataloader(
        type_path=args.type_path, batch_size=args.bs, shuffle=False
    )
    Path(args.output_dir).mkdir(exist_ok=True)
    model.eval()

    # Generate & compute loss
    print(f"Generating with {args.device}...")
    model = model.to(args.device)
    preds, gen_lls, ppl = generate_from_model(data_loader, model, generate=args.generate)

    # also calculate loss for masked graph objective
    if args.save_sentence_losses and prev_args.amr_masking is not None:
        mask_lls = estimate_scaffolding_loss(
            model,
            args.type_path, 
            shuffling=original_shuffling,
            masking=prev_args.amr_masking,
            reordering=None,
            bs=args.bs,
            n_samples=5,
        )
    if args.save_sentence_losses and prev_args.amr_reordering is not None:
        reorder_lls = estimate_scaffolding_loss(
            model,
            args.type_path,
            shuffling=original_shuffling,
            masking=None,
            reordering=prev_args.amr_reordering,
            bs=args.bs,
            n_samples=5,
        )
    
    if args.shuffle_graph_components: # append "shuffled" if shuffling
        args.type_path = f"{args.type_path}-shuffled"
    if args.amr_shuffling:
        args.type_path = f"{args.type_path}-{args.amr_shuffling}" 
    if args.append_second_amr:
        args.type_path = f"{args.type_path}-second_{args.append_second_amr}"
    if args.data_dir is not None: # implies a different evaluation dataset
        args.type_path = f"{Path(args.data_dir).name}-{args.type_path}"

    pickle_save(args, Path(args.output_dir, f"{args.type_path}-hparams.pkl"))

    if args.save_sentence_losses:
        np.save(Path(args.output_dir, f"{args.type_path}-gen_lls.npy"), gen_lls)
        if prev_args.amr_masking is not None:
            np.save(Path(args.output_dir, f"{args.type_path}-mask_lls.npy"), mask_lls)
        if prev_args.amr_reordering is not None:
            np.save(Path(args.output_dir, f"{args.type_path}-reorder_lls.npy"), reorder_lls)

    if not args.generate:
        return None

    with open(Path(args.output_dir, f"{args.type_path}.pred"), "w") as outfile:
        for sent in preds:
            outfile.write(f"{sent}\n")
    if not args.reference_paths:
        return

    # Read in references
    reference_lns = []
    for path in args.reference_paths:
        if not Path(path).exists():
            # silently try in the input directory
            path = Path(prev_args.data_dir, path)

        with open(path, "r") as infile:
            lines = [x.strip() for i, x in enumerate(infile) if i < len(preds)]
        reference_lns.append(lines)

    # Compute scores
    score_fn = calculate_bleu # TODO: support others
    scores = score_fn(preds, reference_lns)
    scores['ppl'] = ppl
    save_json(scores, Path(args.output_dir, f"{args.type_path}-bleu.json"))

    # Create beaker metrics
    full_metrics = {
        "bleu":  scores['bleu'],
        "bleu_normed": scores['bleu_normed'],
        "ppl": ppl,
        "preds": preds,
    }
    full_metrics.update(**training_metrics)
    save_json(full_metrics, Path(args.output_dir, "metrics.json"))


if __name__ == "__main__":
    # Usage for MT:
    # python run_eval.py MODEL_NAME $DATA_DIR/test.source $save_dir/test_translations.txt --reference_path $DATA_DIR/test.target --score_path $save_dir/test_bleu.json  --task translation $@
    run_generate(verbose=True)
