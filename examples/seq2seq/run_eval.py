import argparse
import json
import sys
from pathlib import Path

sys.path.append("..")

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer, T5Tokenizer

from finetune import SummarizationModule, TranslationModule, TranslationModule

try:
    from .utils import pickle_load, pickle_save, save_json, Seq2SeqDataset, calculate_bleu_score
except ImportError:
    from utils import pickle_load, pickle_save, save_json, Seq2SeqDataset, calculate_bleu_score

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_from_model(
    data_loader,
    model,
    device=DEFAULT_DEVICE,
    **gen_kwargs,
) -> None:
    
    all_preds = []
    for batch in tqdm(data_loader):
        pad_token_id = model.tokenizer.pad_token_id
        source_ids, source_mask, y = Seq2SeqDataset.trim_seq2seq_batch(batch, pad_token_id)

        generated_ids = model.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            use_cache=True,
            decoder_start_token_id=model.decoder_start_token_id,
            **gen_kwargs,
        )
        preds = model.ids_to_clean_text(generated_ids)
        all_preds.extend(preds)
        # TODO: add loss to output?
    
    return all_preds

def val_tokenize(lines, tokenizer=None):
    """
    To ensure consistency with other models, we want to tokenize/detokenize in the same
    way across all models.

    TODO: Fix this across all modules, since it's messy
    """
    if tokenizer is None:
        return lines
    
    lines = [" ".join(tokenizer.tokenize(l)) for l in lines]
    if isinstance(tokenizer, T5Tokenizer):
        lines = [l.replace(" ", "").replace("▁", " ") for l in lines]
    if isinstance(tokenizer, BertTokenizer):
        lines = [l.replace(" ##", "") for l in lines]
    return lines


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str, help="Directory containing model and saved args")
    parser.add_argument("output_dir", type=str, help="where to save the outputs")

    parser.add_argument("--type_path", type=str, default="val", help="Split from which to generate, i.e. <type_path>.source")
    parser.add_argument("--reference_paths", nargs="+", required=False, help="like val.target val-2.target")
    parser.add_argument("--data_dir", type=str, help="Input location (default uses that in saved args)")
    parser.add_argument("--tokenizer_path_or_name", default=None, help="Use this to match tokenization between models")
    parser.add_argument("--device", type=str, required=False, default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--bs", type=int, default=8, required=False, help="batch size")
    parser.add_argument(
        "--n_obs", type=int, default=-1, required=False, help="How many observations. Defaults to all."
    )

    args = parser.parse_args()
    
    # Reset previously used arguments as necessary
    prev_args = pickle_load(Path(args.model_dir, "hparams.pkl"))
    prev_args.model_name_or_path = str(Path(prev_args.output_dir, "best_tfmr"))
    setattr(prev_args, f"n_{args.type_path}", args.n_obs)

    if args.data_dir is not None:
        prev_args.data_dir = args.data_dir

    if prev_args.task == "summarization":
        model = SummarizationModule(prev_args)
    elif prev_args.task == "translation":
        model = TranslationModule(prev_args)
    elif prev_args.task == "data-to-text":
        model = TranslationModule(prev_args)

    data_loader = model.get_dataloader(
        type_path=args.type_path, batch_size=args.bs, shuffle=False
    )

    Path(args.output_dir).mkdir(exist_ok=True)

    # Make generations
    print("Generating...")
    preds = generate_from_model(
        data_loader=data_loader,
        model=model,
        output_dir=args.output_dir,
        device=args.device,
    )

    pickle_save(args, Path(args.output_dir, f"{args.type_path}-hparams.pkl"))

    with open(Path(args.output_dir, f"{args.type_path}.pred"), "w") as outfile:
        for sent in preds:
            outfile.write(f"{sent}\n")
    
    if not args.reference_paths:
        return
    
    # Compute scores
    score_fn = calculate_bleu_score # TODO: support others

    val_tokenizer = None
    if args.tokenizer_path_or_name is not None:
        val_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path_or_name)

    preds = val_tokenize(preds, val_tokenizer)

    reference_lns = []
    for path in args.reference_paths:
        if not Path(path).exists():
            # silently try in the input directory
            path = Path(prev_args.data_dir, path)
        
        with open(path, "r") as infile:
            lines = [x.strip() for i, x in enumerate(infile) if i < len(preds)]
        lines = val_tokenize(lines, val_tokenizer)
        reference_lns.append(lines)

    scores = score_fn(preds, reference_lns)
    save_json(scores, Path(args.output_dir, f"{args.type_path}-bleu.json"))


if __name__ == "__main__":
    run_generate()
