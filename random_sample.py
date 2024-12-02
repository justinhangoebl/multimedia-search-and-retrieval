import pandas as pd
import numpy as np

def random_sample(title, artist, infos, topK=10):
    idx_to_drop = infos[(infos['song'] == title) & (infos['artist'] == artist)].index
    if len(idx_to_drop) == 0:
        #print("Song not found; returning just any random sample")
        return infos.sample(topK)
    return infos.drop(idx_to_drop[0]).sample(topK)