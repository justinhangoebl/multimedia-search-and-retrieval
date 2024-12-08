import pandas as pd
import numpy as np
from tqdm import tqdm
from metrics import *
from measures import *
import warnings
warnings.filterwarnings("ignore")

def tf_idf_rec(title, artist, infos, tf_idf, topK=10):
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
        # Get the tf-idf vector for the current match
        tf_idf_vector = tf_idf[(tf_idf["id"] == match["id"])].values[0][1:]

        # Compute cosine similarity for all songs
        similarities = tf_idf.apply(lambda x: cosine_similarity(tf_idf_vector, x[1:]), axis=1)

        # Get the top K most similar songs (excluding the current match)
        topK_similarities = similarities.nlargest(topK + 1)[1:]
        topK_indexes = topK_similarities.index
        topK_songs = tf_idf.iloc[topK_indexes, :]
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

def all_tf_idf(infos, tf_idf, topK=10):
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for TF-IDF processing.")

    def process_song(song):
        rec = tf_idf_rec(song["song"], song["artist"], infos, tf_idf, topK)
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values
        row = np.zeros(len(infos))
        row[infos_idx] = sims
        return row

    with tqdm_joblib(desc="Processing TF-IDF Recommendations", total=len(infos)):
        recs = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
    return np.array(recs)
        sims = rec["sim"].values
        recs[idx, infos_idx] = sims
    return recs
