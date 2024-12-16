import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import cosine_similarity
import os


def preprocess_data(genres, tags, ivec256, ivec512, ivec1024):
    genres_dict = {row["id"]: set(eval(row["genre"])) for _, row in genres.iterrows()}
    tags_dict = {row["id"]: eval(row["(tag, weight)"]) for _, row in tags.iterrows()}
    ivec_dict = {
        row["id"]: np.concatenate([
            ivec256[ivec256["id"] == row["id"]].iloc[:, 1:].values[0],
            ivec512[ivec512["id"] == row["id"]].iloc[:, 1:].values[0],
            ivec1024[ivec1024["id"] == row["id"]].iloc[:, 1:].values[0],
        ])
        for _, row in ivec256.iterrows()
    }
    return genres_dict, tags_dict, ivec_dict


def compute_similarity(song_id, genre_vector, tag_vector, ivec_vector, genres_dict, tags_dict, ivec_dict, alpha, beta, gamma):
    genre_similarity = len(genre_vector & genres_dict.get(song_id, set()))
    tag_similarity = sum(
        min(tags_dict.get(song_id, {}).get(tag, 0), weight) for tag, weight in tag_vector.items()
    )
    ivec_combined_vector = ivec_dict.get(song_id, np.zeros_like(ivec_vector))
    ivec_similarity = cosine_similarity(
        ivec_vector.reshape(1, -1), ivec_combined_vector.reshape(1, -1)
    )[0][0]
    return alpha * genre_similarity + beta * tag_similarity + gamma * ivec_similarity


def feature_aware_recommendation(title, artist, infos, genres_dict, tags_dict, ivec_dict, topK=10, alpha=1.0, beta=1.0, gamma=1.0):
    if title and not artist:
        matches = infos[infos["song"] == title]
    elif artist and not title:
        matches = infos[infos["artist"] == artist]
    else:
        matches = infos[(infos["song"] == title) & (infos["artist"] == artist)]

    if matches.empty:
        return pd.DataFrame({"id": [], "infos_idx": [], "title": [], "artist": [], "sim": []})

    recommendations = []
    for _, match in matches.iterrows():
        song_id = match["id"]
        genre_vector = genres_dict.get(song_id, set())
        tag_vector = tags_dict.get(song_id, {})
        ivec_vector = ivec_dict.get(song_id, np.zeros(len(ivec_dict[next(iter(ivec_dict))])))

        similarities = infos["id"].apply(
            lambda id_: compute_similarity(id_, genre_vector, tag_vector, ivec_vector, genres_dict, tags_dict, ivec_dict, alpha, beta, gamma)
        )

        topK_similarities = similarities.nlargest(topK + 1)[1:]
        topK_indexes = topK_similarities.index
        topK_ids = infos.iloc[topK_indexes]["id"].values

        for idx, id_ in enumerate(topK_ids):
            similar_song = infos[infos["id"] == id_]
            recommendations.append(
                {
                    "id": id_,
                    "infos_idx": similar_song.index[0],
                    "title": similar_song["song"].values[0],
                    "artist": similar_song["artist"].values[0],
                    "sim": topK_similarities.values[idx],
                }
            )

    return pd.DataFrame(recommendations)


def all_feature_aware_recs(infos, genres, tags, ivec256, ivec512, ivec1024, topK=10, alpha=1.0, beta=1.0, gamma=1.0):
    genres_dict, tags_dict, ivec_dict = preprocess_data(genres, tags, ivec256, ivec512, ivec1024)

    n_jobs = max(1, os.cpu_count() // 2)
    print(f"Using {n_jobs} cores for feature-aware recommendation generation.")

    def process_song(song):
        source_id = song["id"]
        rec = feature_aware_recommendation(
            song["song"], song["artist"], infos, genres_dict, tags_dict, ivec_dict, topK, alpha, beta, gamma
        )
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values

        recommendations = [
            {"source_id": source_id, "target_id": infos.iloc[idx]["id"], "similarity": sim}
            for idx, sim in zip(infos_idx, sims)
        ]
        return recommendations

    with tqdm_joblib(desc="Processing Feature-Aware Recommendations", total=len(infos)):
        recs = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())

        all_recommendations = [rec for rec_group in recs for rec in rec_group]
        return pd.DataFrame(all_recommendations)




