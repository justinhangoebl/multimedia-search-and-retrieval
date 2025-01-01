import numpy as np
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import os
import numpy as np
from experimental_metrics import *
from similarity_measures import *
import warnings
warnings.filterwarnings("ignore")

def random_sample(title, artist, infos, topK=10):
    idx_to_drop = infos[(infos['song'] == title) & (infos['artist'] == artist)].index
    if len(idx_to_drop) == 0:
        return infos.sample(topK)
    return infos.drop(idx_to_drop[0]).sample(topK)

def all_random(infos, topK=10):
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for Random Recommendations.")

    def process_song(song):
        ret = random_sample(song["song"], song["artist"], infos, topK)
        row = np.zeros(len(infos))
        row[list(ret.index)] = 1
        return row

    with tqdm_joblib(desc="Processing Random Recommendations", total=len(infos)):
        rets = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
    return np.array(rets)




