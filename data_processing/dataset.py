import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.utils import download_from_url, extract_archive
import io

url_base = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
train_urls = ("train.de.gz", "train.en.gz")
val_urls = ("val.de.gz", "val.en.gz")
test_urls = ("test_2016_flickr.de.gz", "test_2016_flickr.en.gz")

train_filepaths = [
    extract_archive(download_from_url(url_base + url))[0] for url in train_urls
]
val_filepaths = [
    extract_archive(download_from_url(url_base + url))[0] for url in val_urls
]
test_filepaths = [
    extract_archive(download_from_url(url_base + url))[0] for url in test_urls
]

de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

def yield_tokens(filepath, tokenizer):
    with io.open(filepath, encoding="utf8") as f:
        for string_ in f:
            yield tokenizer(string_)

specials = ["<unk>", "<pad>", "<bos>", "<eos>"]
de_vocab = build_vocab_from_iterator(yield_tokens(train_filepaths[0], de_tokenizer), specials=specials, special_first=True)
de_vocab.set_default_index(de_vocab["<unk>"])

en_vocab = build_vocab_from_iterator(yield_tokens(train_filepaths[1], en_tokenizer), specials=specials, special_first=True)
en_vocab.set_default_index(en_vocab["<unk>"])

def data_process(filepaths):
    raw_de_iter = iter(io.open(filepaths[0], encoding="utf8"))
    raw_en_iter = iter(io.open(filepaths[1], encoding="utf8"))
    data = []
    for raw_de, raw_en in zip(raw_de_iter, raw_en_iter):
        de_tensor_ = torch.tensor(
            [de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long
        )
        en_tensor_ = torch.tensor(
            [en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long
        )
        data.append((de_tensor_, en_tensor_))
    return data

train_data = data_process(train_filepaths)
val_data = data_process(val_filepaths)
test_data = data_process(test_filepaths)
