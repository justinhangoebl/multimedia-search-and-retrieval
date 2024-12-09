import gradio as gr
from tf_idf import *
from bert import *
from random_sample import *
from w2v import *
from vgg import *
from resnet import *
from incp import *
from ivec import *
import pandas as pd

infos = pd.read_csv("dataset/id_information_mmsr.tsv", sep="\t")
original_infos = infos.copy()

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
            if len(partial_matches_combined) == 1:
                single_match = partial_matches_combined.iloc[0]
                match_song = single_match["song"]
                match_artist = single_match["artist"]
                match_details = f"Did you mean:\n\n{combined_suggestions}\n\n"
                if model == "ivec":
                    model_data1 = datasets["ivec256"]()
                    model_data2 = datasets["ivec512"]()
                    model_data3 = datasets["ivec1024"]()
                    recs = ivec_rec(match_song, match_artist, original_infos, model_data1, model_data2, model_data3,
                                    topK=Amount)
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
                    else:
                        recs = random_sample(match_song, match_artist, original_infos, topK=Amount)
                else:
                    return f"Invalid model: {model}. Please select a valid model."

                recs = recs.merge(original_infos, on="id", suffixes=("_match", ""))
                recommendations = recs[["song", "artist", "sim"]].to_markdown(index=False)
                return match_details + "\nRecommendations:\n\n" + recommendations
            return f"Did you mean:\n\n{combined_suggestions}"

        partial_matches_song = pd.DataFrame()
        partial_matches_artist = pd.DataFrame()

        if Song:
            partial_matches_song = original_infos[original_infos["song"].str.contains(Song, case=False)]
        if Artist:
            partial_matches_artist = original_infos[original_infos["artist"].str.contains(Artist, case=False)]

        if partial_matches_song.empty and partial_matches_artist.empty:
            return "No matches found."

        suggestions = ""
        if not partial_matches_song.empty:
            song_suggestions = partial_matches_song[["song", "artist"]].to_markdown(index=False)
            suggestions += f"Songs matching '{Song}':\n\n{song_suggestions}\n\n"

        if not partial_matches_artist.empty:
            artist_suggestions = partial_matches_artist[["artist"]].drop_duplicates().to_markdown(index=False)
            suggestions += f"Artists matching '{Artist}':\n\n{artist_suggestions}\n\n"

        return f"Did you mean:\n\n{suggestions.strip()}"

    output = ""
    match_count = 0

    for _, match in matches.iterrows():
        match_song = original_infos.loc[match.name, "song"]
        match_artist = original_infos.loc[match.name, "artist"]

        match_details = f"### Match {match_count + 1}\n**Song:** {match_song}\n\n**Artist:** {match_artist}\n\n"

        if model == "ivec":
            model_data1 = datasets["ivec256"]()
            model_data2 = datasets["ivec512"]()
            model_data3 = datasets["ivec1024"]()
            recs = ivec_rec(match_song, match_artist, original_infos, model_data1, model_data2, model_data3,
                            topK=Amount)
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
            else:
                recs = random_sample(match_song, match_artist, original_infos, topK=Amount)
        else:
            return f"Invalid model: {model}. Please select a valid model."

        recs = recs.merge(original_infos, on="id", suffixes=("_match", ""))
        recommendations = recs[["song", "artist", "sim"]].to_markdown(index=False)
        match_count += 1

        output += match_details + "\n" + recommendations + "\n\n---\n\n"

    return output.strip("\n---\n")


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
            ["Bert", "Random", "TF-IDF", "Word2Vec", "VGG19", "ResNet", "Inception", "ivec"],
            label="Model",
            value="Bert",
        ),
    ],
    outputs=gr.Markdown(),
    live=False,
    css=custom_css,
)

demo.launch()
