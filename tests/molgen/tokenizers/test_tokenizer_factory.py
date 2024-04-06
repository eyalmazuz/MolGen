import json
import tempfile

import pytest

from molgen.tokenizers.tokenizer_factory import get_tokenizer

@pytest.fixture
def tempdir():
    tempdir = tempfile.TemporaryDirectory()
    yield tempdir
    tempdir.cleanup()


@pytest.fixture
def token_to_id():
    tokens_to_ids = {"<pad>": 0, "<s>": 1, "</s>": 2, "A": 3, "B": 4, "C": 5}
    return tokens_to_ids


@pytest.fixture
def char_config():
    config = {"type": "CharTokenizer",
              "kwargs": {"model_max_length": 16,
                         "special_tokens": None}}
    return config


@pytest.fixture
def char_tokenizer(tempdir, token_to_id, char_config):
    with open(f"{tempdir.name}/config.json", "w") as f:
        json.dump(char_config, f)

    with open(f"{tempdir.name}/vocab.json", "w") as f:
        json.dump(token_to_id, f)


def test_valid_char_tokenizer(tempdir, char_tokenizer, token_to_id):
    tokenizer = get_tokenizer(tempdir.name)
    
    assert tokenizer.tokens_to_ids == token_to_id
    assert tokenizer.model_max_length == 16


def test_invalid_path():
    with pytest.raises(ValueError):
        tokenizer = get_tokenizer("/foo/bar")


def test_no_config(tempdir):
    with pytest.raises(ValueError):
        tokenizer = get_tokenizer(tempdir.name)

