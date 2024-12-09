import gradio as gr
from tf_idf import *
from bert import *
from random_sample import *
from w2v import *
from vgg import *
from resnet import *
from incp import *
from ivec import *
from f_a import *
import pandas as pd

infos = pd.read_csv("dataset/id_information_mmsr.tsv", sep="\t")
original_infos = infos.copy()
url_data = pd.read_csv("dataset/id_url_mmsr.tsv", sep="\t")

datasets = {
    "TF-IDF": lambda: pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep="\t"),
    "Bert": lambda: pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep="\t"),
    "Word2Vec": lambda: pd.read_csv("dataset/id_lyrics_word2vec_mmsr.tsv", sep="\t"),
    "VGG19": lambda: pd.read_csv("dataset/id_vgg19_mmsr.tsv", sep="\t"),
    "ResNet": lambda: pd.read_csv("dataset/id_resnet_mmsr.tsv", sep="\t"),
    "Inception": lambda: pd.read_csv("dataset/id_incp_mmsr.tsv", sep="\t"),
    "ivec256": lambda: pd.read_csv("dataset/id_ivec256_mmsr.tsv", sep="\t"),
    "ivec512": lambda: pd.read_csv("dataset/id_ivec512_mmsr.tsv", sep="\t"),
    "ivec1024": lambda: pd.read_csv("dataset/id_ivec1024_mmsr.tsv", sep="\t"),
}


def embed_links(recs):
    recs = recs.merge(url_data, on="id", how="left")
    recs["song_with_link"] = recs.apply(
        lambda row: f"[{row['song'][:30]}...]({row['url']})" if len(row['song']) > 30 else f"[{row['song']}]({row['url']})",
        axis=1
    )
    return recs


def greet(Song, Artist, Amount, model):
    infos["song"] = infos["song"].str.lower()
    infos["artist"] = infos["artist"].str.lower()

    if Song:
        Song = Song.lower()
    if Artist:
        Artist = Artist.lower()

    if Song and Artist:
        matches = infos[(infos["song"] == Song) & (infos["artist"] == Artist)]
    elif Song:
        matches = infos[infos["song"] == Song]
    elif Artist:
        matches = infos[infos["artist"] == Artist]
    else:
        return "Please provide at least a song or artist name."

    if matches.empty:
        partial_matches_combined = pd.DataFrame()
        if Song and Artist:
            partial_matches_combined = original_infos[
                (original_infos['song'].str.contains(Song, case=False)) &
                (original_infos['artist'].str.contains(Artist, case=False))
            ]

        if len(partial_matches_combined) > 0:
            combined_suggestions = partial_matches_combined[['song', 'artist']].to_markdown(index=False)
            return f"Did you mean:\n\n{combined_suggestions}"

        return "No matches found."

    output = ""
    genres = pd.read_csv("dataset/id_genres_mmsr.tsv", sep="\t")
    tags = pd.read_csv("dataset/id_tags_dict.tsv", sep="\t")
    ivec256 = datasets["ivec256"]()
    ivec512 = datasets["ivec512"]()
    ivec1024 = datasets["ivec1024"]()
    genres_dict, tags_dict, ivec_dict = preprocess_data(genres, tags, ivec256, ivec512, ivec1024)

    for _, match in matches.iterrows():
        match_song = original_infos.loc[match.name, "song"]
        match_artist = original_infos.loc[match.name, "artist"]

        match_details = f"**Song:** {match_song}\n**Artist:** {match_artist}\n\n"

        if model == "ivec":
            model_data1 = datasets["ivec256"]()
            model_data2 = datasets["ivec512"]()
            model_data3 = datasets["ivec1024"]()
            recs = ivec_rec(match_song, match_artist, original_infos, model_data1, model_data2, model_data3, topK=Amount)
        elif model == "f_a ivec":
            recs = []
            match_id = match["id"]
            genre_vector = genres_dict.get(match_id, set())
            tag_vector = tags_dict.get(match_id, {})
            ivec_vector = ivec_dict.get(match_id, np.zeros(len(ivec_dict[next(iter(ivec_dict))])))

            for song_id in infos["id"]:
                if song_id == match_id:
                    continue  # Exclude the song itself
                sim = compute_similarity(song_id, genre_vector, tag_vector, ivec_vector, genres_dict, tags_dict, ivec_dict, 1.0, 1.0, 1.0)
                recs.append((song_id, sim))

            recs = pd.DataFrame(recs, columns=["id", "sim"]).sort_values(by="sim", ascending=False).head(Amount)
        elif model in datasets:
            model_data = datasets[model]()
            if model == "TF-IDF":
                recs = tf_idf_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
            elif model == "Bert":
                recs = bert_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
            elif model == "Word2Vec":
                recs = word2vec_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
            elif model == "VGG19":
                recs = vgg19_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
            elif model == "ResNet":
                recs = resnet_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
            elif model == "Inception":
                recs = inception_rec(match_song, match_artist, original_infos, model_data, topK=Amount)
        elif model == "Random":
                recs = random_sample(match_song, match_artist, original_infos, topK=Amount)
        else:
            return f"Invalid model: {model}. Please select a valid model."

        recs = recs.merge(original_infos, on="id", suffixes=("_match", ""))
        recs = embed_links(recs)  # Add links to recommendations
        if model != "Random":
            recommendations = recs[["song_with_link", "artist", "sim"]].to_markdown(index=False)
            recommendations = recommendations.replace("song_with_link", "Songs")
            recommendations = recommendations.replace("artist", "Artist")
            recommendations = recommendations.replace("sim", "Similarity")
        else:
            recommendations = recs[["song_with_link", "artist"]].to_markdown(index=False)
            recommendations = recommendations.replace("song_with_link", "Songs")
            recommendations = recommendations.replace("artist", "Artist")
        # rename song_with_link to songs

        output += match_details + "Recommendations:\n\n" + recommendations + "\n\n"

    return output.strip()


custom_css = """
#root {
    max-width: 100%;
    margin: auto;
}
.markdown-body {
    font-size: 16px;
    line-height: 1.5;
}
"""

demo = gr.Interface(
    fn=greet,
    inputs=[
        "text",
        "text",
        gr.Slider(value=10, minimum=1, maximum=50, step=1, label="Number of Recommendations"),
        gr.Radio(
            ["Bert", "Random", "TF-IDF", "Word2Vec", "VGG19", "ResNet", "Inception", "ivec", "f_a ivec"],
            label="Model",
            value="Bert",
        ),
    ],
    outputs=gr.Markdown(),
    live=False,
    css=custom_css,
)

demo.launch()

