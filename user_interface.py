import gradio as gr
from tf_idf import *
from random_sample import *
import pandas as pd

def greet(Song, Artist, Amount, model):
    infos = pd.read_csv("dataset/id_information_mmsr.tsv", sep="\t")
    if model == "TF-IDF":
        tf_idf = pd.read_csv("dataset/id_lyrics_tf-idf_mmsr.tsv", sep="\t")
        recs = tf_idf_rec(Song, Artist, infos, tf_idf, topK=Amount)
        return recs[['title', 'artist', 'sim']]
    elif model == "Bert":
        bert = pd.read_csv("dataset/id_lyrics_bert_mmsr.tsv", sep="\t")
        recs = tf_idf_rec(Song, Artist, infos, bert, topK=Amount)
        return recs[['title', 'artist', 'sim']]
    elif model == "Random":
        recs = random_sample(Song, Artist, infos, topK=Amount)
        return recs[['song', 'artist']]
    return pd.DataFrame({"title": [], "artist": [], "sim": []})

demo = gr.Interface(
    fn=greet,
    inputs=["text", "text"],
    additional_inputs=[gr.Slider(value=10, minimum=1, maximum=50, step=1),
                       gr.Radio(["TF-IDF", "Random", "Bert"], label="Model", value="TF-IDF")],
    outputs=[gr.DataFrame(label="Recommendations")],
)
demo.launch()
