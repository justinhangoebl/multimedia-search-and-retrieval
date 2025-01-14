import pandas as pd
import numpy as np
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import os
import numpy as np
from experimental_metrics import *
from similarity_measures import *
import warnings
warnings.filterwarnings("ignore")

def find_matches(title, artist, infos):
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

    return matches

def vector_based_retrieval(title, artist, infos, song_vector_tuple, similarity_fn=cosine_similarity, topK=10):
    matches = find_matches(title, artist, infos)
    if matches.empty:
        return pd.DataFrame({"id": [], "infos_idx": [], "title": [], "artist": [], "sim": []})

    # Prepare to collect retrieval for each match
    retrieved = []

    for _, match in matches.iterrows():
        # Get the BERT vector for the current match
        vector = song_vector_tuple[(song_vector_tuple["id"] == match["id"])].values[0][1:].astype(float)

        # Compute cosine similarity for all songs
        similarities = song_vector_tuple.iloc[:, 1:].apply(
            lambda x: similarity_fn(vector, x.astype(float).values), axis=1
        )

        # Get the top K most similar songs (excluding the current match)
        topK_similarities = similarities.nlargest(topK + 1)[1:]
        topK_indexes = topK_similarities.index
        topK_songs = song_vector_tuple.iloc[topK_indexes, :]
        topK_ids = topK_songs["id"].values

        # Append results for the current match
        for idx, id in enumerate(topK_ids):
            similar_song = infos[infos["id"] == id]
            retrieved.append(
                {
                    "id": id,
                    "infos_idx": similar_song.index[0],
                    "title": similar_song["song"].values[0],
                    "artist": similar_song["artist"].values[0],
                    "sim": topK_similarities.values[idx],
                }
            )

    # Return all collected recommendations as a DataFrame
    return pd.DataFrame(retrieved)

def generate_single_modal_vector_retrieval(infos, bert, similarity_fn=cosine_similarity, topK=10):
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for Single Modal Vector processing.")

    def process_song(song):
        ret = vector_based_retrieval(song["song"], song["artist"], infos, bert, similarity_fn, topK)
        infos_idx = ret["infos_idx"].values
        sims = ret["sim"].values
        source_id = song["id"]  # Get the source song ID
        recommendations = [
            {"source_id": source_id, "target_id": infos.iloc[idx]["id"], "similarity": sim}
            for idx, sim in zip(infos_idx, sims)
        ]
        return recommendations

    all_retrieved = []
    with tqdm_joblib(desc="Processing Single Modal Vector Retrieval", total=len(infos)):
        results = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
        for rets in results:
            all_retrieved.extend(rets)

    # Convert to a DataFrame for saving and further use
    rets_df = pd.DataFrame(all_retrieved)
    return rets_df

def generate_scores_matrix_for_sm_vectors(infos, bert, similarity_fn=cosine_similarity, topK=10):
    n_jobs = max(1, os.cpu_count()//2)
    print(f"Using {n_jobs} cores for Single Modal Vector processing.")

    def process_song(song):
        ret = vector_based_retrieval(song["song"], song["artist"], infos, bert, similarity_fn, topK)
        infos_idx = ret["infos_idx"].values
        sims = ret["sim"].values
        row = np.zeros(len(infos))
        row[infos_idx] = sims
        return row

    with tqdm_joblib(desc="Processing Single Modal Vector Retrieval", total=len(infos)):
        rets = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
    return np.array(rets)
