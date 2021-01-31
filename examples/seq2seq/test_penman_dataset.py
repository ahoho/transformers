from pathlib import Path
from itertools import product 
import unittest

from transformers import AutoTokenizer
from penman_denoising import AMR_SOURCES, AMR_TARGETS, AMR_NOISED
from utils import PenmanDataset, trim_batch


def _dump_articles(path: Path, articles: list):
    content = "\n".join(articles)
    Path(path).write_text(content)


def make_test_data_dir(tmp_dir):
    for split in ["train", "val", "test"]:
        _dump_articles(Path(tmp_dir, f"{split}.source"), AMR_SOURCES)
        _dump_articles(Path(tmp_dir, f"{split}.target"), AMR_TARGETS)
    return tmp_dir


class TestPenman(unittest.TestCase):
    def test_denoising(self):
        tokenizer = AutoTokenizer.from_pretrained("t5-large")
        Path("tmp").mkdir(exist_ok=True)
        make_test_data_dir("./tmp")

        shuffling = [None, "reconfigure", "randomize"]
        masking_components = [None, "all", "components", "nodes", "surface"]
        masking_style = [None, "mass", "drop", "corrupt"]
        reordering = [
            None,
            "reorder",
            "parse-from-triples",
            "generate-from-triples",
            "convert-to-triples",
        ]
        
        # Construct the settings --- not all combinations are logical, so we need to correct
        all_settings = {}
        combins = product(shuffling, masking_components, masking_style, reordering)
        for shuff, mask, mask_style, reorder in combins:
            dataset_kwargs = {}
            description = []
            
            # Indicates whether to shuffle the linearization
            if shuff and mask != "surface":
                dataset_kwargs["graph_shuffling"] = shuff
                description.append(shuff)

            # Do we include a masking task?
            if mask:
                if mask in ["all", "surface"]:
                    prob = 0.15
                if mask == "components":
                    prob = 0.21 #~15%, since 70% of graph tokens are structuralall_
                if mask == "nodes":
                    prob = 0.5  #~15%
                dataset_kwargs["graph_token_masking_prob"] = prob
                
                if mask_style == "corrupt" and mask != "components":
                    continue # currently doesn't make sense for "components" or "nodes"
                if mask_style and mask != "surface":
                    mask += f"-{mask_style}"

                if reorder and shuff and mask_style and mask != "surface" and mask_style != "drop":
                    mask += "-unshuffle"
                
                dataset_kwargs["graph_masking"] = mask
                description.append(f"mask-{mask}")

            # Do we do a reordering task?
            if reorder and not mask and (shuff or "triples" in reorder):
                dataset_kwargs["graph_reordering"] = reorder
                description.append(reorder)
            
            if dataset_kwargs:
                description = "_".join(description)
                all_settings[description] = dataset_kwargs
        
        for description, kwargs in sorted(all_settings.items()):
            dataset = PenmanDataset(
                tokenizer=tokenizer,
                data_dir="./tmp",
                graph_masking_mixture=1.0,
                max_source_length=512,
                max_target_length=512,
                type_path="val",
                eval_seed=31415,
                shuffle_eval=True,
                **kwargs,
            )
            src, tgt = dataset.__getitem__(0, return_text=True)
            reference = AMR_NOISED[description]
            self.assertEqual(reference["src"], src)
            self.assertEqual(reference["tgt"], tgt)

if __name__ == "__main__":
    unittest.main()