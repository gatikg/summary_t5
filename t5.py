from transformers import pipeline
from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize


# dataset = load_dataset("cnn_dailymail", version="3.0.0")
# sample_text = dataset["train"][1]["article"][:1000]

def t5_test(sample_text):
    nltk.download('punkt')
    summaries = {}
    pipe = pipeline('summarization', model='t5-small')
    pipe_out = pipe(sample_text)
    summaries['t5'] = 'n'.join(sent_tokenize(pipe_out[0]['summary_text']))
    return summaries['t5'], sample_text, len(sample_text), len(summaries['t5'])
