import gradio as gr
from tf_idf import *
from bert import *
from random_sample import *
import pandas as pd

def greet(Song, Artist, Amount, model):
    infos = pd.read_csv("dataset/id_information_mmsr.tsv", sep="\t")
    original_infos = infos.copy()

    if Song:
        Song = Song.lower()
        infos['song'] = infos['song'].str.lower()
    if Artist:
        Artist = Artist.lower()
        infos['artist'] = infos['artist'].str.lower()

    if model == "TF-IDF":
        tf_idf = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep="\t")
        if Song and Artist:
            recs = tf_idf_rec(Song, Artist, infos, tf_idf, topK=Amount)
        elif Song:
            recs = tf_idf_rec(Song, None, infos, tf_idf, topK=Amount)
        elif Artist:
            recs = tf_idf_rec(None, Artist, infos, tf_idf, topK=Amount)
        else:
            return pd.DataFrame({"error": ["Please provide at least a song or artist name."]})

        recs = recs.merge(original_infos, on='id', suffixes=('_match', ''))
        return recs[['song', 'artist', 'sim']]
    elif model == "Bert":
        bert = pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep="\t")
        if Song and Artist:
            recs = bert_rec(Song, Artist, infos, bert, topK=Amount)
        elif Song:
            recs = bert_rec(Song, None, infos, bert, topK=Amount)
        elif Artist:
            recs = bert_rec(None, Artist, infos, bert, topK=Amount)
        else:
            return pd.DataFrame({"error": ["Please provide at least a song or artist name."]})

        recs = recs.merge(original_infos, on='id', suffixes=('_match', ''))
        return recs[['song', 'artist', 'sim']]
    elif model == "Random":
        recs = random_sample(Song, Artist, infos, topK=Amount)
        recs = recs.merge(original_infos, on='id', suffixes=('_match', ''))
        return recs[['song', 'artist']]
    return pd.DataFrame({"error": ["Invalid request."]})

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text"],
    additional_inputs=[gr.Slider(value=10, minimum=1, maximum=50, step=1),
                       gr.Radio(["TF-IDF", "Random", "Bert"], label="Model", value="TF-IDF")],
    outputs=[gr.DataFrame(label="Recommendations")],
)
demo.launch()
