from dash import Dash, dcc, html, Input, Output, State, ctx
import pandas as pd
from tf_idf import *
from bert import *
from random_sample import *
from w2v import *
from vgg import *
from resnet import *
from incp import *
from ivec import *
from f_a import *
from pyngrok import ngrok, conf
import subprocess

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

models = ["Bert", "Random", "TF-IDF", "Word2Vec", "VGG19", "ResNet", "Inception", "ivec", "f_a ivec"]

app = Dash(__name__)


def embed_links(recs):
    recs = recs.merge(url_data, on="id", how="left")
    recs["song_with_link"] = recs.apply(
        lambda row: f"{row['song']}" if len(row['song']) <= 30 else f"{row['song'][:30]}...",
        axis=1,
    )
    recs["url"] = recs["url"]
    return recs


app.layout = html.Div(
    style={"display": "flex", "flexDirection": "row", "height": "100vh", "padding": "20px"},
    children=[
        html.Div(
            style={"width": "50%", "paddingRight": "20px"},
            children=[
                html.H2("Music Recommendation System", style={"textAlign": "center"}),

                html.Label("Enter a Song Name:"),
                dcc.Input(id="song-input", type="text", placeholder="Enter song name",
                          style={"width": "100%", "marginBottom": "15px"}),

                html.Label("Enter an Artist Name:"),
                dcc.Input(id="artist-input", type="text", placeholder="Enter artist name",
                          style={"width": "100%", "marginBottom": "15px"}),

                html.Label("Number of Recommendations:"),
                html.Div(
                    style={"marginBottom": "30px"},
                    children=[
                        dcc.Slider(
                            id="amount-slider",
                            min=1,
                            max=50,
                            step=1,
                            value=10,
                            marks={1: "1", 25: "25", 50: "50"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        )
                    ],
                ),

                html.Label("Select a Model:"),
                dcc.RadioItems(
                    id="model-dropdown",
                    options=[{"label": m, "value": m} for m in models],
                    value="Bert",
                    inline=True,
                    style={"marginBottom": "30px"},
                ),

                html.Label("Select an Artist or Song"),
                dcc.Dropdown(
                    id="dropdown-container",
                    placeholder="Select an option",
                    style={"width": "100%", "marginBottom": "20px"},
                ),

                html.Div(
                    style={"display": "flex", "justifyContent": "space-between"},
                    children=[
                        html.Button("Clear", id="clear-button",
                                    style={"backgroundColor": "#d6d6d6", "padding": "10px 20px"}),
                        html.Button("Submit", id="recommend-button",
                                    style={"backgroundColor": "#ff6600", "padding": "10px 20px", "color": "white"}),
                    ],
                ),
            ],
        ),
        html.Div(
            style={"width": "50%", "backgroundColor": "#f6f6f6", "padding": "20px", "borderRadius": "10px"},
            children=[
                html.H3("Recommendations", style={"textAlign": "center", "marginBottom": "20px"}),
                html.Div(id="recommendations", style={"overflowY": "auto", "maxHeight": "80vh"}),
            ],
        ),
    ],
)


@app.callback(
    Output("dropdown-container", "options"),
    Output("dropdown-container", "value"),
    Output("dropdown-container", "style"),
    Input("song-input", "value"),
    Input("artist-input", "value"),
)
def update_dropdown(song, artist):
    infos["song"] = infos["song"].str.lower()
    infos["artist"] = infos["artist"].str.lower()

    if song:
        song_lower = song.lower()
    if artist:
        artist_lower = artist.lower()

    exact_options = []
    partial_options = []

    if song and not artist:
        exact_matches = infos[infos["song"] == song_lower]
        if len(exact_matches) > 1:
            exact_options = [{"label": a, "value": a} for a in
                             original_infos.loc[exact_matches.index, "artist"].unique()]

        partial_matches = original_infos[
            (original_infos['song'].str.contains(song_lower, case=False, na=False, regex=False))
        ]
        if not exact_matches.empty:
            partial_matches = partial_matches[~partial_matches.index.isin(exact_matches.index)]

        if not partial_matches.empty:
            partial_options = [{"label": f"{row['artist']} (Partial Match)", "value": row['artist']} for _, row in
                               partial_matches.iterrows()]

    elif artist and not song:
        exact_matches = infos[infos["artist"] == artist_lower]
        if len(exact_matches) > 1:
            exact_options = [{"label": s, "value": s} for s in original_infos.loc[exact_matches.index, "song"].unique()]

        partial_matches = original_infos[
            (original_infos['artist'].str.contains(artist_lower, case=False, na=False, regex=False))
        ]
        if not exact_matches.empty:
            partial_matches = partial_matches[~partial_matches.index.isin(exact_matches.index)]

        if not partial_matches.empty:
            partial_options = [{"label": f"{row['song']} (Partial Match)", "value": row['song']} for _, row in
                               partial_matches.iterrows()]

    options = exact_options + partial_options
    if options:
        return options, None, {"display": "block"}
    else:
        return [], None, {"display": "none"}


@app.callback(
    Output("recommend-button", "n_clicks"),
    Input("dropdown-container", "value"),
)
def trigger_submit_button(dropdown_value):
    if dropdown_value:
        return 1
    return 0


@app.callback(
    Output("recommendations", "children"),
    [Input("recommend-button", "n_clicks"),
     Input("song-input", "n_submit"),
     Input("artist-input", "n_submit")],
    [State("song-input", "value"),
     State("artist-input", "value"),
     State("amount-slider", "value"),
     State("model-dropdown", "value"),
     State("dropdown-container", "value")],
)
def recommend(n_clicks, n_submit_song, n_submit_artist, song, artist, amount, model, dropdown_value):
    if not n_clicks and not n_submit_song and not n_submit_artist:
        return html.Div("No recommendations yet.")

    infos["song"] = infos["song"].str.lower()
    infos["artist"] = infos["artist"].str.lower()

    if song:
        song = song.lower()
    if artist:
        artist = artist.lower()

    if dropdown_value:
        if song and not artist:
            artist = dropdown_value.lower()
        elif artist and not song:
            song = dropdown_value.lower()

    if song and artist:
        matches = infos[(infos["song"] == song) & (infos["artist"] == artist)]
    elif song:
        matches = infos[infos["song"] == song]
    elif artist:
        matches = infos[infos["artist"] == artist]
    else:
        return html.Div("Please provide at least a song or artist name.")

    if matches.empty:
        if song and artist:
            partial_matches = original_infos[
                original_infos['song'].str.contains(song, case=False, na=False, regex=False) &
                original_infos['artist'].str.contains(artist, case=False, na=False, regex=False)
                ]
        elif song:
            partial_matches = original_infos[
                original_infos['song'].str.contains(song, case=False, na=False, regex=False)
            ]
        elif artist:
            partial_matches = original_infos[
                original_infos['artist'].str.contains(artist, case=False, na=False, regex=False)
            ]

        if partial_matches.empty:
            return html.Div("No matches found.", style={"color": "red"})

        matches = partial_matches.copy()

    match = matches.iloc[0]
    match_song = original_infos.loc[match.name, "song"]
    match_artist = original_infos.loc[match.name, "artist"]

    if model == "ivec":
        model_data1 = datasets["ivec256"]()
        model_data2 = datasets["ivec512"]()
        model_data3 = datasets["ivec1024"]()
        recs = ivec_rec(match_song, match_artist, original_infos, model_data1, model_data2, model_data3, topK=amount)
    elif model == "f_a ivec":
        genres = pd.read_csv("dataset/id_genres_mmsr.tsv", sep="\t")
        tags = pd.read_csv("dataset/id_tags_dict.tsv", sep="\t")
        ivec256 = datasets["ivec256"]()
        ivec512 = datasets["ivec512"]()
        ivec1024 = datasets["ivec1024"]()
        genres_dict, tags_dict, ivec_dict = preprocess_data(genres, tags, ivec256, ivec512, ivec1024)

        match_id = match["id"]
        genre_vector = genres_dict.get(match_id, set())
        tag_vector = tags_dict.get(match_id, {})
        ivec_vector = ivec_dict.get(match_id, np.zeros(len(ivec_dict[next(iter(ivec_dict))])))

        recs = []
        for song_id in infos["id"]:
            if song_id == match_id:
                continue
            sim = compute_similarity(
                song_id, genre_vector, tag_vector, ivec_vector, genres_dict, tags_dict, ivec_dict, 1.0, 1.0, 1.0
            )
            recs.append((song_id, sim))

        recs = pd.DataFrame(recs, columns=["id", "sim"]).sort_values(by="sim", ascending=False).head(amount)

    elif model == "Random":
        recs = random_sample(match_song, match_artist, original_infos, topK=amount)
    else:
        model_data = datasets[model]()
        if model == "TF-IDF":
            recs = tf_idf_rec(match_song, match_artist, original_infos, model_data, topK=amount)
        elif model == "Bert":
            recs = bert_rec(match_song, match_artist, original_infos, model_data, topK=amount)
        elif model == "Word2Vec":
            recs = word2vec_rec(match_song, match_artist, original_infos, model_data, topK=amount)
        elif model == "VGG19":
            recs = vgg19_rec(match_song, match_artist, original_infos, model_data, topK=amount)
        elif model == "ResNet":
            recs = resnet_rec(match_song, match_artist, original_infos, model_data, topK=amount)
        elif model == "Inception":
            recs = inception_rec(match_song, match_artist, original_infos, model_data, topK=amount)

    recs = recs.merge(original_infos, on="id", suffixes=("_match", ""))
    recs = embed_links(recs)

    recs.rename(columns={"song_with_link": "Songs", "artist": "Artist", "sim": "Similarity"}, inplace=True)

    return html.Div([
        html.H4("Match 1"),
        html.P(f"Song: {match_song}"),
        html.P(f"Artist: {match_artist}"),
        html.Table(
            [html.Tr([html.Th("Song"), html.Th("Artist"), html.Th("Similarity")])] +
            [
                html.Tr([
                    html.Td(html.A(row["Songs"], href=row["url"], target="_blank")),
                    html.Td(row["Artist"]),
                    html.Td(f"{row['Similarity']:.2f}")
                ])
                for _, row in recs.iterrows()
            ],
            style={"width": "100%", "border": "1px solid black", "borderCollapse": "collapse"},
        ),
    ])


if __name__ == "__main__":
    subprocess.call(["taskkill", "/F", "/IM", "ngrok.exe"])
    conf.get_default().auth_token = "2q5OFT3nILP0FGlDX3kIqKMu43c_3JbjoqxPMHhFVFPYafK4c"
    public_url = ngrok.connect(8050)
    print(f"Public URL: {public_url}")
    app.run_server(debug=True, port=8050)
