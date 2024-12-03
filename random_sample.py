import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from metrics import *
from measures import *

def random_sample(title, artist, infos, topK=10):
    idx_to_drop = infos[(infos['song'] == title) & (infos['artist'] == artist)].index
    if len(idx_to_drop) == 0:
        return infos.sample(topK)
    return infos.drop(idx_to_drop[0]).sample(topK)

def all_random_recs(infos,  topK=10):
    recs = np.zeros((len(infos), len(infos)))
    for idx, song in tqdm(infos.iterrows(), total=infos.shape[0]):
        rec = random_sample(song["song"], song["artist"], infos, topK)
        recs[idx, list(rec.index)] = 1
    return recs
