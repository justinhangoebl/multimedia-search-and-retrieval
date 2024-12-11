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

def feature_aware_recommendation(title, artist, infos, genres_dict, tags_dict, ivec_dict, explored_set, topK=10, alpha=1.0, beta=1.0, gamma=1.0, novelty_weight=0.5):
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

        # Compute genre similarity
        genre_similarities = infos["id"].apply(lambda id_: len(genre_vector & genres_dict.get(id_, set())))

        # Compute tag similarity
        tag_similarities = infos["id"].apply(
            lambda id_: sum(min(tags_dict.get(id_, {}).get(tag, 0), weight) for tag, weight in tag_vector.items())
        )

        # Compute i-vector cosine similarity
        ivec_vectors = np.array([ivec_dict.get(id_, np.zeros_like(ivec_vector)) for id_ in infos["id"]])
        ivec_similarities = cosine_similarity(ivec_vector.reshape(1, -1), ivec_vectors).flatten()

        # Compute novelty scores
        if explored_set:
            mean_explored_vector = np.mean([ivec_dict[e] for e in explored_set], axis=0)
            novelty_scores = 1.0 - cosine_similarity(mean_explored_vector.reshape(1, -1), ivec_vectors).flatten()
        else:
            novelty_scores = np.ones(len(infos))

        novelty_scores = np.maximum(novelty_scores, 0.1) ** 2  # Penalize repeated items

        # Combine similarities
        similarities = (
            alpha * genre_similarities +
            beta * tag_similarities +
            gamma * ivec_similarities +
            novelty_weight * novelty_scores
        )
        similarities = pd.Series(similarities, index=infos["id"])  # Explicitly align with infos["id"]

        # Select top K recommendations
        topK_similarities = similarities.nlargest(topK + 1).iloc[1:]  # Exclude the current song
        topK_ids = topK_similarities.index

        for id_ in topK_ids:
            similar_song = infos[infos["id"] == id_]
            recommendations.append(
                {
                    "id": id_,
                    "infos_idx": similar_song.index[0],  # Keep original index intact
                    "title": similar_song["song"].values[0],
                    "artist": similar_song["artist"].values[0],
                    "sim": topK_similarities.loc[id_],
                }
            )
            explored_set.add(id_)  # Dynamically update explored set

    return pd.DataFrame(recommendations)




def all_feature_aware_recs_novel(infos, genres, tags, ivec256, ivec512, ivec1024, topK=10, alpha=1.0, beta=1.0, gamma=1.0, novelty_weight=0.5):
    genres_dict, tags_dict, ivec_dict = preprocess_data(genres, tags, ivec256, ivec512, ivec1024)

    n_jobs = max(1, os.cpu_count() // 2 + os.cpu_count() // 4)
    print(f"Using {n_jobs} cores for feature-aware recommendation generation.")

    def process_song(song):
        explored_set = set()
        rec = feature_aware_recommendation(
            song["song"], song["artist"], infos, genres_dict, tags_dict, ivec_dict, explored_set, topK, alpha, beta, gamma, novelty_weight
        )
        infos_idx = rec["infos_idx"].values
        sims = rec["sim"].values
        row = np.zeros(len(infos))
        row[infos_idx] = sims
        return row

    with tqdm_joblib(desc="Processing Feature-Aware Recommendations", total=len(infos)):
        recs = Parallel(n_jobs=n_jobs)(delayed(process_song)(song) for _, song in infos.iterrows())
    return np.array(recs)
