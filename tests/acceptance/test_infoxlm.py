import pytest
from torch.testing import assert_close
from transformers import AutoTokenizer, AutoModelForTextEncoding

from transformer_lens import HookedEncoder

MODEL_NAME = "microsoft/infoxlm-large"


@pytest.fixture(scope="module")
def our_infoxlm():
    return HookedEncoder.from_pretrained(MODEL_NAME, device="cpu")


@pytest.fixture(scope="module")
def huggingface_infoxlm():
    return AutoModelForTextEncoding.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture
def hello_world_tokens(tokenizer):
    return tokenizer("Hello, world!", return_tensors="pt")["input_ids"]


def test_full_model(our_infoxlm, huggingface_infoxlm, tokenizer):
    sequences = [
        "Hello, world!",
        "this is another sequence of tokens",
    ]
    tokenized = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    hf_out = huggingface_infoxlm(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hf_embedding = hf_out.pooler_output
    hf_cache = hf_out.hidden_states
    our_embedding, our_cache = our_infoxlm.run_with_cache(input_ids, one_zero_attention_mask=attention_mask)

    assert_close(hf_embedding, our_embedding, rtol=1.3e-6, atol=4e-5)
    assert_close(our_cache['blocks.23.hook_normalized_resid_post'], hf_cache[-1], rtol=1.3e-6, atol=4e-5)
