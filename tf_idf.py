import pandas as pd
import numpy as np
from tqdm import tqdm
import ast
from metrics import *
from measures import *

def tf_idf_rec(title, artist, infos, tf_idf, topK=10):
    # Get the index of the song
    song = infos[(infos["song"]==title) & (infos["artist"]==artist)]
    # Get the tf-idf vector of the song
    tf_idf_vector = tf_idf[(tf_idf["id"]==song["id"].values[0])].values[0][1:]
    # Compute the cosine similarity between the song and all the others
    similarities = tf_idf.apply(lambda x: cosine_similarity(tf_idf_vector, x[1:]), axis=1)
    # Get the top K most similar songs
    topK_similarities = similarities.nlargest(topK+1)[1:]
    # Get the index of the top K most similar songs
    topK_indexes = topK_similarities.index
    # Get the title and artist of the top K most similar songs
    topK_songs = tf_idf.iloc[topK_indexes, :]
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

def all_tf_idf(infos, tf_idf, topK=10):
    recs = np.zeros((len(infos), len(infos)))
    for idx, song in tqdm(infos.iterrows(), total=infos.shape[0]):
        rec = tf_idf_rec(song["song"], song["artist"], infos, tf_idf, topK)
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values
        recs[idx, infos_idx] = sims
    return recs