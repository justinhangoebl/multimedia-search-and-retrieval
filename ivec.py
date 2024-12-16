import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings

warnings.filterwarnings("ignore")


def ivec_rec(title, artist, infos, ivec256, ivec512, ivec1024, topK=10):
    if title and not artist:
        matches = infos[infos["song"] == title]
    elif artist and not title:
        matches = infos[infos["artist"] == artist]
    else:
        matches = infos[(infos["song"] == title) & (infos["artist"] == artist)]

    if matches.empty:
        return pd.DataFrame({"id": [], "infos_idx": [], "title": [], "artist": [], "sim": []})

    recommendations = []

    for _, match in tqdm(matches.iterrows(), total=len(matches), desc="Processing Matches"):
        try:
            ivec256_vector = ivec256[ivec256["id"] == match["id"]].iloc[:, 1:].values[0].astype(float)
            ivec512_vector = ivec512[ivec512["id"] == match["id"]].iloc[:, 1:].values[0].astype(float)
            ivec1024_vector = ivec1024[ivec1024["id"] == match["id"]].iloc[:, 1:].values[0].astype(float)

            combined_ivec_vector = np.hstack([ivec256_vector, ivec512_vector, ivec1024_vector])

            similarities = ivec256.iloc[:, 1:].apply(
                lambda row: cosine_similarity(
                    [combined_ivec_vector],
                    [np.hstack([
                        row.values.astype(float),
                        ivec512.iloc[row.name, 1:].values.astype(float),
                        ivec1024.iloc[row.name, 1:].values.astype(float),
                    ])])[0][0],
                axis=1
            )

            topK_similarities = similarities.nlargest(topK + 1)[1:]
            topK_indexes = topK_similarities.index
            topK_songs = ivec256.iloc[topK_indexes, :]
            topK_ids = topK_songs["id"].values

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
        except Exception as e:
            print(f"Error processing match {match['id']}: {e}")

    return pd.DataFrame(recommendations)


def all_ivec_recs(infos, ivec256, ivec512, ivec1024, topK=10):
    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for iVector recommendation processing.")

    def process_song(song):
        source_id = song["id"]
        rec = ivec_rec(song["song"], song["artist"], infos, ivec256, ivec512, ivec1024, topK)
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values

        recommendations = [
            {"source_id": source_id, "target_id": infos.iloc[idx]["id"], "similarity": sim}
            for idx, sim in zip(infos_idx, sims)
        ]
        return recommendations

    with tqdm_joblib(desc="Processing iVector Recommendations", total=len(infos)):
        recs = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())

        all_recommendations = [rec for rec_group in recs for rec in rec_group]
        return pd.DataFrame(all_recommendations)
