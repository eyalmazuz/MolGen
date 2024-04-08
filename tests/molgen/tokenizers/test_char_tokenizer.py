import json
import tempfile

import pytest
import torch

from molgen.tokenizers.char_tokenizer import CharTokenizer


@pytest.fixture
def token_to_id():
    tokens_to_ids = {"<pad>": 0, "<s>": 1, "</s>": 2, "A": 3, "B": 4, "C": 5}
    return tokens_to_ids


@pytest.fixture
def tempdir():
    tempdir = tempfile.TemporaryDirectory()
    yield tempdir
    tempdir.cleanup()

def test_load_pretrained_invalid_path():
    with pytest.raises(ValueError):
        CharTokenizer.load_pretrained(path="/foo/bar")


def test_load_pretrained_empty_path(tempdir):
    with pytest.raises(ValueError):
        CharTokenizer.load_pretrained(path=tempdir.name)


def test_load_pretrained_valid_path(tempdir, token_to_id):
    with open(f"{tempdir.name}/vocab.json", "w") as f:
        json.dump(token_to_id, f)

    tokenizer = CharTokenizer.load_pretrained(path=tempdir.name)
    
    assert token_to_id == tokenizer.tokens_to_ids
    assert {v: k for k, v in token_to_id.items()} == tokenizer.ids_to_tokens


def test_load_pretrained_valid_path_and_config(tempdir, token_to_id):
    with open(f"{tempdir.name}/vocab.json", "w") as f:
        json.dump(token_to_id, f)

    tokenizer = CharTokenizer.load_pretrained(path=tempdir.name)
    
    assert token_to_id == tokenizer.tokens_to_ids
    assert {v: k for k, v in token_to_id.items()} == tokenizer.ids_to_tokens


def test_encode(token_to_id):
    tokenizer = CharTokenizer(token_to_id, special_tokens=["<s>", "</s>", "<pad>"])

    encoding = tokenizer.encode("ACB")
    assert encoding == [[3,5,4]]

    encoding = tokenizer.encode("<s>ACB")
    assert encoding == [[6,3,5,4]]

    encoding = tokenizer.encode("ACB</s>")
    assert encoding == [[3,5,4,7]]

    encoding = tokenizer.encode("<s>ACB</s>")
    assert encoding == [[6,3,5,4,7]]

    encoding = tokenizer.encode("ACB", return_tensors=True)
    assert torch.all(encoding == torch.tensor([[3,5,4]]))


def test_encode_padding(token_to_id):
    tokenizer = CharTokenizer(token_to_id, special_tokens=["<s>", "</s>", "<pad>"])

    encoding = tokenizer.encode("ACB", padding="max_length", max_length=6)
    assert encoding == [[3,5,4,8,8,8]]

    encoding = tokenizer.encode("<s>ACB", padding="max_length", max_length=5)
    assert encoding == [[6,3,5,4,8]]

    encoding = tokenizer.encode(["<s>ACB", "<s>CCCC"], padding=True)
    assert encoding == [[6,3,5,4,8], [6,5,5,5,5]]

    encoding = tokenizer.encode(["<s>ACB</s>", "<s>CCCC</s>"], padding=True)
    assert encoding == [[6,3,5,4,7,8], [6,5,5,5,5,7]]

    encoding = tokenizer.encode(["ACB</s>", "CCCC</s>"], padding=True, return_tensors=True)
    assert torch.all(encoding == torch.tensor([[3,5,4,7,8], [5,5,5,5,7]]))


def test_decode(token_to_id):
    tokenizer = CharTokenizer(token_to_id, special_tokens=["<s>",
                                                           "</s>", 
                                                           "<pad>"])

    text = tokenizer.decode([3,5,4,8,8,8])
    assert text == ["ACB<pad><pad><pad>"]

    text = tokenizer.decode([3,5,4,8,8,8], skip_special_tokens=True)
    assert text == ["ACB"]

    text = tokenizer.decode([6,3,5,4])
    assert text == ["<s>ACB"]

    text = tokenizer.decode([6,3,5,4], skip_special_tokens=True)
    assert text == ["ACB"]

    text = tokenizer.decode([6,3,5,4,7], skip_special_tokens=True)
    assert text == ["ACB"]

    text = tokenizer.decode([6,3,5,4,7])
    assert text == ["<s>ACB</s>"]

