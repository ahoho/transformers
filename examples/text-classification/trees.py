import json
from pathlib import Path
import sys
import random

from nltk.corpus import BracketParseCorpusReader
from nltk.tree import Tree


class ShuffleTree(Tree):
    def format_parse(self, nodesep="", parens="()", quotes=False, shuffle=False, drop_label=False, naked_leaves=True):
        childstrs = []
        rand_idx = random.sample(range(len(self)), len(self)) if shuffle else range(len(self))
        for idx in rand_idx:
            child = self[idx]
            if isinstance(child, Tree):
                childstrs.append(
                    child.format_parse(
                        nodesep=nodesep,
                        parens=parens,
                        quotes=quotes,
                        shuffle=shuffle,
                        drop_label=drop_label,
                        naked_leaves=naked_leaves
                    )
                )
            elif isinstance(child, tuple):
                childstrs.append("/".join(child))
            elif isinstance(child, str) and not quotes:
                childstrs.append("%s" % child)
            else:
                childstrs.append(repr(child))
        if naked_leaves and len(childstrs) == 1: # a leaf
            parens = ("", "")
        if isinstance(self._label, str):
            return "%s%s%s %s%s" % (
                parens[0],
                self._label if not drop_label else "",
                nodesep,
                " ".join(childstrs),
                parens[1],
            )
        else:
            return "%s%s%s %s%s" % (
                parens[0],
                repr(self._label) if not drop_label else "",
                nodesep,
                " ".join(childstrs),
                parens[1],
            )


class ShuffleTreeCorpusReader(BracketParseCorpusReader):
    def _parse(self, t):
        try:
            tree = ShuffleTree.fromstring(self._normalize(t))
            # If there's an empty node at the top, strip it off
            if tree.label() == '' and len(tree) == 1:
                return tree[0]
            else:
                return tree

        except ValueError as e:
            sys.stderr.write("Bad tree detected; trying to recover...\n")
            # Try to recover, if we can:
            if e.args == ("mismatched parens",):
                for n in range(1, 5):
                    try:
                        v = ShuffleTree(self._normalize(t + ")" * n))
                        sys.stderr.write(
                            "  Recovered by adding %d close " "paren(s)\n" % n
                        )
                        return v
                    except ValueError:
                        pass
            # Try something else:
            sys.stderr.write("  Recovered by returning a flat parse.\n")
            # sys.stderr.write(' '.join(t.split())+'\n')
            return ShuffleTree("S", self._tag(t))


if __name__ == "__main__":
    for fpath in Path("./trees").glob("*.txt"):
        sst_data = ShuffleTreeCorpusReader("./trees", fpath.name)
        sst_parsed = [
            json.dumps({
                "text": t.format_parse(shuffle=False, drop_label=False),
                "idx": i,
                "label": int(t.label()),
                "label_reg": float(t.label()),
                "label_3cl": {0: 1, 1:1, 2:2, 3:3, 4:3}[int(t.label())],
            })
            for i, t in enumerate(sst_data.parsed_sents())
        ]
        with open(f"./trees/{fpath.stem}.json", "w") as outfile:
            outfile.write("\n".join(sst_parsed))
        if fpath.stem == "train":
            random.seed(42)
            random.shuffle(sst_parsed)
            with open(f"./trees/{fpath.stem}-1000.json", "w") as outfile:
                outfile.write("\n".join(sst_parsed[:1000]))