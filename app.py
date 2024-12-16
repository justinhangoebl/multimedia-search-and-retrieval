import gradio as gr
import pandas as pd
import numpy as np
from random_sample import random_sample

infos = pd.read_csv("dataset/id_information_mmsr.tsv.gz", sep="\t", compression="gzip")
original_infos = infos.copy()
url_data = pd.read_csv("dataset/id_url_mmsr.tsv.gz", sep="\t", compression="gzip")

datasets = {
    "TF-IDF": None,
    "Bert": None,
    "Word2Vec": None,
    "VGG19": None,
    "ResNet": None,
    "Inception": None,
    "ivec": None,
    "Feature-Aware ivec": None,
    "MFCC BoW": None,
    "MFCC Stats": None,
}

recommendation_files = {
    "TF-IDF": "predictions/recs_tf_idf_10.csv",
    "Bert": "predictions/recs_bert_10.csv",
    "Word2Vec": "predictions/recs_word2vec_10.csv",
    "VGG19": "predictions/recs_vgg19_10.csv",
    "ResNet": "predictions/recs_resnet_10.csv",
    "Inception": "predictions/recs_incp_10.csv",
    "ivec": "predictions/recs_ivec_10.csv",
    "Feature-Aware ivec": "predictions/recs_f_a_10.csv",
    "MFCC BoW": "predictions/recs_mfcc_bow_10.csv",
    "MFCC Stats": "predictions/recs_mfcc_stats_10.csv",
}

def rename_columns(df):
    rename_mapping = {
        "song_with_link": "Song",
        "artist": "Artist",
        "sim": "Similarity"
    }
    return df.rename(columns=rename_mapping)

def embed_links(recs):
    recs = recs.merge(url_data, on="id", how="left")
    recs["song_with_link"] = recs.apply(
        lambda row: (
            f"[{row['song'][:30]}...]({row['url']})"
            if len(row['song']) > 30 else f"[{row['song']}]({row['url']})"
        ), axis=1
    )
    return recs

def load_recommendation_file(model):
    if datasets[model] is None and recommendation_files.get(model):
        datasets[model] = pd.read_csv(recommendation_files[model])

def get_recommendations(song_id, model, amount, match_song, match_artist):
    if model == "Random":
        return random_sample(match_song, match_artist, original_infos, topK=amount)
    load_recommendation_file(model)
    recs = datasets.get(model)
    if recs is None:
        return pd.DataFrame()
    recs = recs[recs["source_id"] == song_id].nlargest(amount, "similarity")
    recs = recs.merge(original_infos, left_on="target_id", right_on="id", suffixes=("_source", ""))
    recs = recs.rename(columns={"similarity": "Similarity"})
    return recs

def greet(Song, Artist, Amount, model, dropdown_value=None):
    infos["song"] = infos["song"].str.lower()
    infos["artist"] = infos["artist"].str.lower()

    if dropdown_value and " - " in dropdown_value:
        Song, Artist = dropdown_value.split(" - ")

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
        return "Please provide at least a song or artist name.", gr.update(visible=False), gr.update(visible=False), "", ""

    if matches.empty:
        partial_matches = original_infos[
            (original_infos['song'].str.contains(Song, case=False)) &
            (original_infos['artist'].str.contains(Artist, case=False))
        ] if Song and Artist else original_infos[
            (original_infos['song'].str.contains(Song, case=False)) |
            (original_infos['artist'].str.contains(Artist, case=False))
        ]

        if len(partial_matches) == 1:
            match = partial_matches.iloc[0]
            Song, Artist = match["song"], match["artist"]
            matches = infos[(infos["song"] == Song.lower()) & (infos["artist"] == Artist.lower())]
        else:
            if len(partial_matches) > 0:
                combined_suggestions = partial_matches[['song', 'artist']].to_markdown(index=False)
                return f"Did you mean:\n\n{combined_suggestions}", gr.update(visible=False), gr.update(visible=False), "", ""
            return "No matches found.", gr.update(visible=False), gr.update(visible=False), "", ""

    output = ""

    for _, match in matches.iterrows():
        match_song = original_infos.loc[match.name, "song"]
        match_artist = original_infos.loc[match.name, "artist"]
        song_url = url_data[url_data["id"] == match["id"]]["url"].values[0]

        match_details = f"### Song: [{match_song}]({song_url}) by {match_artist}\n\n"

        recs = get_recommendations(match["id"], model, Amount, match_song, match_artist)
        recs = embed_links(recs)
        recs = rename_columns(recs)

        if "Similarity" in recs.columns:
            recommendations = recs[["Song", "Artist", "Similarity"]].to_markdown(index=False)
        else:
            recommendations = recs[["Song", "Artist"]].to_markdown(index=False)

        output += match_details + "Recommendations:\n\n" + recommendations + "\n\n"

    if len(matches) == 1:
        return output.strip(), gr.update(visible=True), gr.update(visible=True), matches.iloc[0]["song"].lower(), matches.iloc[0]["artist"].lower()
    else:
        return output.strip(), gr.update(visible=False), gr.update(visible=False), "", ""


def compare_recommendations(song, artist, models_to_compare, amount):
    if not song or not artist:
        return "No track selected."

    match = infos[(infos["song"] == song) & (infos["artist"] == artist)]
    if match.empty:
        return "No valid track selected for comparison."
    match = match.iloc[0]

    original_song = original_infos.loc[match.name, "song"]
    original_artist = original_infos.loc[match.name, "artist"]
    song_url = url_data[url_data["id"] == match["id"]]["url"].values[0]

    output = f"### Song: [{original_song}]({song_url}) by {original_artist}\n\n"

    for model in models_to_compare:
        if model != "Random":
            load_recommendation_file(model)

    model_recommendations = {}
    model_dataframes = {}

    for model in models_to_compare:
        if model == "Random":
            recs = random_sample(original_song, original_artist, original_infos, topK=amount)
        else:
            recs = get_recommendations(match["id"], model, amount, original_song, original_artist)

        if not recs.empty:
            recs = embed_links(recs)
            recs = rename_columns(recs)
            model_dataframes[model] = recs

            if "Similarity" in recs.columns:
                recommendations = recs[["Song", "Artist", "Similarity"]].to_markdown(index=False)
            else:
                recommendations = recs[["Song", "Artist"]].to_markdown(index=False)
        else:
            recommendations = f"No recommendations available for model {model}."

        model_recommendations[model] = recommendations

    if len(models_to_compare) > 1:
        non_random_models = [m for m in models_to_compare if m != "Random"]
        if len(non_random_models) > 1:
            common_ids = set.intersection(
                *(set(df["id"]) for model, df in model_dataframes.items() if model != "Random" and not df.empty))
            if common_ids:
                common_recs = original_infos[original_infos["id"].isin(common_ids)].copy()
                common_recs = embed_links(common_recs)
                common_recs = rename_columns(common_recs)
                common_table = common_recs[["Song", "Artist"]].to_markdown(index=False)
                output += "### Common Recommendations Across All Non-Random Models\n\n"
                output += common_table + "\n\n---\n\n"

    output += "<div style='display: flex; flex-wrap: wrap; gap: 10px;'>\n"
    for model, table in model_recommendations.items():
        output += f"<div style='flex: 1 1 calc(50% - 20px); padding: 10px; border: 1px solid #ccc;'>"
        output += f"<h2><b>{model} Recommendations</b></h2>\n\n{table}</div>\n"
    output += "</div>\n"

    return output

custom_css = """
#root {
    max-width: 100%;
    margin: auto;
    display: flex;
    flex-direction: row;
}
.left-panel {
    width: 50%;
    padding-right: 20px;
}
.right-panel {
    width: 50%;
    background-color: #f6f6f6;
    padding: 20px;
    border-radius: 10px;
}
#recommend-button {
    background-color: #ff6600;
    color: white;
    border: none;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border-radius: 5px;
}
#recommend-button:hover {
    background-color: #ff8533;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column(elem_id="left-panel"):
            song_input = gr.Textbox(label="Enter Song Name", interactive=True)
            artist_input = gr.Textbox(label="Enter Artist Name", interactive=True)
            dropdown_choices = gr.Dropdown(label="Multiple Matches, Please select one:", interactive=True, visible=False)
            exact_song_state = gr.State()
            exact_artist_state = gr.State()
            amount_slider = gr.Slider(value=10, minimum=1, maximum=100, step=1, label="Number of Recommendations")
            model_selection = gr.Radio(
                [
                    "Random", "Bert", "TF-IDF", "Word2Vec",
                    "MFCC BoW", "MFCC Stats", "ivec",
                    "VGG19", "ResNet", "Inception",
                    "Feature-Aware ivec"
                ],
                label="Model",
                value="Bert",
            )

            recommend_button = gr.Button("Recommend", elem_id="recommend-button")
            roll_dice_button = gr.Button("I'm feeling (un)lucky", visible=True)

            compare_models = gr.CheckboxGroup(
                [
                    "Random", "Bert", "TF-IDF", "Word2Vec",
                    "MFCC BoW", "MFCC Stats", "ivec",
                    "VGG19", "ResNet", "Inception",
                    "Feature-Aware ivec"
                ],
                label="Models to Compare (up to 4)",
                visible=False
            )

            compare_button = gr.Button("Compare", visible=False)

        with gr.Column(elem_id="right-panel"):
            recommendations_output = gr.Markdown()

    def handle_update(song, artist, amount, model):
        infos["song"] = infos["song"].str.lower()
        infos["artist"] = infos["artist"].str.lower()

        if song and artist:
            exact_matches = original_infos[
                (original_infos["song"].str.lower() == song.lower()) &
                (original_infos["artist"].str.lower() == artist.lower())
            ]
            partial_matches = original_infos[
                (original_infos["song"].str.contains(song, case=False)) &
                (original_infos["artist"].str.contains(artist, case=False))
            ]
        elif song:
            exact_matches = original_infos[original_infos["song"].str.lower() == song.lower()]
            partial_matches = original_infos[original_infos["song"].str.contains(song, case=False)]
        elif artist:
            exact_matches = original_infos[original_infos["artist"].str.lower() == artist.lower()]
            partial_matches = original_infos[original_infos["artist"].str.contains(artist, case=False)]
        else:
            exact_matches = pd.DataFrame()
            partial_matches = pd.DataFrame()

        matches = pd.concat([exact_matches, partial_matches]).drop_duplicates(subset=["song", "artist"])

        if len(matches) == 1:
            match = matches.iloc[0]
            res, cb_update, cm_update, es, ea = greet(match["song"], match["artist"], amount, model)
            return gr.update(choices=[], value=None, visible=False), res, cb_update, cm_update, es, ea

        if not matches.empty:
            dropdown_options = [f"{row['song']} - {row['artist']}" for _, row in matches.iterrows()]
            return gr.update(choices=dropdown_options, value=None, visible=True), "", gr.update(visible=False), gr.update(visible=False), "", ""

        return gr.update(choices=[], value=None, visible=False), "No matches found.", gr.update(visible=False), gr.update(visible=False), "", ""

    song_input.change(
        handle_update,
        inputs=[song_input, artist_input, amount_slider, model_selection],
        outputs=[dropdown_choices, recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

    artist_input.change(
        handle_update,
        inputs=[song_input, artist_input, amount_slider, model_selection],
        outputs=[dropdown_choices, recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

    dropdown_choices.change(
        greet,
        inputs=[song_input, artist_input, amount_slider, model_selection, dropdown_choices],
        outputs=[recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

    def handle_model_change(song, artist, amount, model, dropdown_value, es, ea):
        if dropdown_value and " - " in dropdown_value:
            s, a = dropdown_value.split(" - ")
            return greet(s, a, amount, model, None)
        elif es and ea:
            return greet(es, ea, amount, model, None)
        else:
            return "", gr.update(visible=False), gr.update(visible=False), "", ""

    model_selection.change(
        handle_model_change,
        inputs=[song_input, artist_input, amount_slider, model_selection, dropdown_choices, exact_song_state, exact_artist_state],
        outputs=[recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

    recommend_button.click(
        greet,
        inputs=[song_input, artist_input, amount_slider, model_selection, dropdown_choices],
        outputs=[recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

    def handle_compare(song, artist, models, amount):
        if len(models) == 1:
            return "Please select more than 1 model for comparison."
        if len(models) > 4:
            return "Please select up to 4 models."
        if len(models) == 0:
            return "No models selected for comparison."
        return compare_recommendations(song, artist, models, amount)

    compare_button.click(
        handle_compare,
        inputs=[exact_song_state, exact_artist_state, compare_models, amount_slider],
        outputs=recommendations_output
    )


    def roll_dice(amount, model):
        random_match = infos.sample(1).iloc[0]
        song = random_match["song"]
        artist = random_match["artist"]
        return greet(song, artist, amount, model)


    def toggle_roll_dice_visibility(song, artist):
        return gr.update(visible=not song and not artist)


    demo.load(
        toggle_roll_dice_visibility,
        inputs=[song_input, artist_input],
        outputs=[roll_dice_button],
    )

    song_input.change(
        toggle_roll_dice_visibility,
        inputs=[song_input, artist_input],
        outputs=[roll_dice_button],
    )

    artist_input.change(
        toggle_roll_dice_visibility,
        inputs=[song_input, artist_input],
        outputs=[roll_dice_button],
    )

    roll_dice_button.click(
        roll_dice,
        inputs=[amount_slider, model_selection],
        outputs=[recommendations_output, compare_button, compare_models, exact_song_state, exact_artist_state],
    )

demo.launch()
