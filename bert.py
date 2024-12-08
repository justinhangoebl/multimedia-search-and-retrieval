import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import *
from measures import *
import warnings
warnings.filterwarnings("ignore")

def bert_rec(title, artist, infos, bert, topK=10):
    # Handle case where only one of `title` or `artist` is provided
    if title and not artist:
        # Find all artists that have a song with the given title
        matches = infos[infos["song"] == title]
    elif artist and not title:
        # Find all songs by the given artist
        matches = infos[infos["artist"] == artist]
    else:
        # Use both `title` and `artist` if provided
        matches = infos[(infos["song"] == title) & (infos["artist"] == artist)]

    if matches.empty:
        return pd.DataFrame({"id": [], "infos_idx": [], "title": [], "artist": [], "sim": []})

    # Prepare to collect recommendations for each match
    recommendations = []

    for _, match in matches.iterrows():
        # Get the BERT vector for the current match
        bert_vector = bert[(bert["id"] == match["id"])].values[0][1:].astype(float)

        # Compute cosine similarity for all songs
        similarities = bert.iloc[:, 1:].apply(
            lambda x: cosine_similarity(bert_vector, x.astype(float).values), axis=1
        )

        # Get the top K most similar songs (excluding the current match)
        topK_similarities = similarities.nlargest(topK + 1)[1:]
        topK_indexes = topK_similarities.index
        topK_songs = bert.iloc[topK_indexes, :]
        topK_ids = topK_songs["id"].values

        # Append results for the current match
        for idx, id in enumerate(topK_ids):
            similar_song = infos[infos["id"] == id]
            recommendations.append(
                {
                    "id": id,
                    "infos_idx": similar_song.index[0],
                    "title": similar_song["song"].values[0],
                    "artist": similar_song["artist"].values[0],
                    "sim": topK_similarities.values[idx],
                }
            )

    # Return all collected recommendations as a DataFrame
    return pd.DataFrame(recommendations)


from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import os
import numpy as np

def all_bert_recs(infos, bert, topK=10):
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for BERT processing.")

    def process_song(song):
        rec = bert_rec(song["song"], song["artist"], infos, bert, topK)
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values
        row = np.zeros(len(infos))
        row[infos_idx] = sims
        return row

    with tqdm_joblib(desc="Processing BERT Recommendations", total=len(infos)):
        recs = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
    return np.array(recs)
