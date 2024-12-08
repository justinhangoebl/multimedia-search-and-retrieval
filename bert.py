import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import *
from measures import *

def bert_rec(title, artist, infos, bert, topK=10):
    
    song = infos[(infos["song"] == title) if title else True &
                                                        (infos["artist"] == artist) if artist else True]
    bert_vector = bert[(bert["id"]==song["id"].values[0])].values[0][1:]
    
    similarities = bert.apply(lambda x: jaccard_similarity(bert_vector, x[1:]), axis=1)
    topK_similarities = similarities.nlargest(topK+1)[1:]

    topK_indexes = topK_similarities.index
    topK_songs = bert.iloc[topK_indexes, :]
    topK_ids = topK_songs["id"].values

    return_dict = []

    for idx, id in enumerate(topK_ids):
        song = infos[infos["id"]==id]
        return_dict.append(
            {
                "id": id,
                "infos_idx": song.index[0],
                "title": song["song"].values[0],
                "artist": song["artist"].values[0],
                'sim': topK_similarities.values[idx]
            }
        )

    return pd.DataFrame(return_dict)

def all_bert_recs(infos, bert, topK=10):
    recs = np.zeros((len(infos), len(infos)))
    for idx, song in tqdm(infos.iterrows(), total=infos.shape[0]):
        rec = bert_rec(song["song"], song["artist"], infos, bert, topK)
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values
        recs[idx, infos_idx] = sims
    return recs
