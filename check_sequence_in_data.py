from pathlib import Path
import numpy as np
import json

data_folder = Path("/home/teusink/code/DiffSBDD/data/pmhc/")
template = "hla_a_{}"

split2suffixes = {
    "0.2" : ["0.2_1", "0.2_2"],
    "0.4" : ["0.4_1", "0.4_2"],
    "full" : ["02_02_9_clustered", "02_02_9_clustered_2", "02_02_9_clustered_3"]
}

split2seq = {}
split2len = {}

for split, suffixes in split2suffixes.items():
    no_unique_peps = []
    for suffix in suffixes:
        unique_peps = set()
        data_path = data_folder / template.format(suffix)
        train_set = np.load(data_path / "train.npz")
        with open(data_path / "decoder.json", "r") as f:
            decoder = json.load(f)

        one_hot = train_set["lig_one_hot"]
        mask = train_set["lig_mask"]
        for i in range(mask[-1] + 1):
            # decode the one-hot
            one_hot_i = one_hot[mask == i]
            
            one_hot_i = np.argmax(one_hot_i, axis=1)
            # use the decoder to get the sequence
            seq = "".join([decoder[j] for j in one_hot_i])
            unique_peps.add(seq)

        no_unique_peps.append(len(unique_peps))
        split2len[split] = mask[-1] + 1

        split2seq[split] = no_unique_peps


print(split2seq)
print(split2len)