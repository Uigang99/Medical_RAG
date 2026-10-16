import copy
import importlib.util
import json
from pathlib import Path

import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from medrag.generation.kg_structure_attention import StructureBias, answer_losses, token_layout
from medrag.training.graph_input_data import SYSTEM
from medrag.text_graph_evaluation import generate_tokens


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "SYS:" + messages[0]["content"] + "USER:" + messages[1]["content"] + "ASSISTANT:"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)), "offset_mapping": [(i, i+1) for i in range(len(text))]}


def row_fixture():
    tok = CharacterTokenizer()
    props = [{"proposition_id": "p1", "claim": "Alpha inhibits Beta."},
             {"proposition_id": "p2", "claim": "Beta activates Gamma."}]
    edges = [{"proposition_id": "p1", "subject_surface": "Alpha", "subject_entity_id": "A",
              "object_surface": "Beta", "object_entity_id": "B"},
             {"proposition_id": "p2", "subject_surface": "Beta", "subject_entity_id": "B",
              "object_surface": "Gamma", "object_entity_id": "C"}]
    row = {"sample_id": "x", "question": "What about Alpha?", "options": {"A": "Beta", "B": "Gamma"},
           "propositions": props, "edges": edges, "answer": "A"}
    user = "Evidence:\n" + "\n".join(p["claim"] for p in props) + "\n\nQuestion:\n" + row["question"] + "\nA. Beta\nB. Gamma"
    text = tok.apply_chat_template([{"content": SYSTEM}, {"content": user}])
    row["text_prompt_ids"] = tok(text)["input_ids"]
    return tok, row, text


def test_evidence_only_identity_direction_and_no_gold_dependence():
    tok, row, text = row_fixture()
    out = token_layout(tok, row)
    pairs = out["pairs"]
    assert out["statistics"]["mapped_edges"] == 2
    assert {t for t, _, _ in pairs} == {0, 1}  # Subject precedes object; reverse cannot see future.
    limit = text.index("\n\nQuestion:")
    assert all(k <= q < limit for _, q, k in pairs)
    assert all(text[q].strip() and text[k].strip() for _, q, k in pairs)
    other = copy.deepcopy(row); other["answer"] = "B"
    assert token_layout(tok, other) == out
    row["edges"].append(copy.deepcopy(row["edges"][0]))
    assert token_layout(tok, row)["pairs"] == pairs  # duplicate triples don't multiply biases


def test_missing_ambiguous_and_conflicting_identity_skip_only_overlay():
    tok, row, _ = row_fixture()
    original = row["text_prompt_ids"][:]
    row["edges"][0]["subject_surface"] = "NOT PRESENT"
    x = token_layout(tok, row)
    assert x["statistics"]["missing_surface"] == 1
    assert row["text_prompt_ids"] == original
    duplicate = copy.deepcopy(row["edges"][1]); duplicate["subject_entity_id"] = "NOT_B"
    row["edges"].append(duplicate)
    x = token_layout(tok, row)
    assert x["statistics"]["identity_ambiguous_mentions"] == 2
    assert x["statistics"]["mapped_edges"] == 2  # two source-grounded edges still usable
    assert not any(t == 0 for t, _, _ in x["pairs"])  # don't merge conflicting identities
    row["text_prompt_ids"][0] += 1
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "prompt identity"):
        token_layout(tok, row)


def tiny_model():
    torch.manual_seed(123)
    torch.set_num_threads(1)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0., bos_token_id=1, eos_token_id=2, pad_token_id=0)
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


def test_zero_bias_matches_logits_padding_cached_and_generation():
    model = tiny_model()
    ids = torch.tensor([[0, 0, 1, 3, 4, 5], [1, 6, 3, 7, 8, 9]])
    mask = (ids != 0).long(); pos = (mask.cumsum(-1)-1).clamp_min(0)
    with torch.no_grad():
        expected = model(ids, attention_mask=mask, position_ids=pos).logits
    prompts = [[1, 3, 4, 5], [1, 6, 3, 7, 8, 9]]
    old = generate_tokens(model, prompts, pad_id=0, eos_ids=[2], max_new_tokens=3)
    bias = StructureBias(2, 4, query_chunk=2)
    layouts = [{"pairs": [[0, 3, 1], [1, 2, 1]]}, {"pairs": [[0, 4, 1], [1, 5, 3]]}]
    bias.set_layouts(layouts, shifts=[2, 0]); bias.attach(model)
    with torch.no_grad():
        result = model(ids, attention_mask=mask, position_ids=pos).logits
    torch.testing.assert_close(result[mask.bool()], expected[mask.bool()], atol=1e-6, rtol=1e-5)
    assert generate_tokens(model, prompts, pad_id=0, eos_ids=[2], max_new_tokens=3) == old
    with torch.no_grad(): bias.raw.fill_(.7)
    with torch.no_grad():
        pref = model(ids, attention_mask=mask, position_ids=pos, use_cache=True)
        appended = torch.cat((ids, torch.ones(2, 1, dtype=torch.long)*10), 1)
        am = torch.cat((mask, torch.ones(2, 1, dtype=torch.long)), 1)
        full = model(appended, attention_mask=am, position_ids=(am.cumsum(-1)-1).clamp_min(0)).logits[:, -1]
        cached = model(appended[:, -1:], attention_mask=am, position_ids=am.sum(-1, keepdim=True)-1,
                       past_key_values=pref.past_key_values, use_cache=True).logits[:, -1]
    torch.testing.assert_close(full, cached, atol=1e-6, rtol=1e-5)


def test_gradient_checkpoint_and_tiny_optimization():
    model = tiny_model()
    bias = StructureBias(2, 4, query_chunk=2); bias.attach(model)
    row = {"text_prompt_ids": [1, 3, 4, 5, 6], "target_ids": [9, 10]}
    layout = {"pairs": [[0, 3, 1], [1, 4, 2]]}
    loss = answer_losses(model, bias, [row], [layout], 0).mean()
    loss.backward(); grad = bias.raw.grad.clone()
    assert grad.abs().sum() > 0 and torch.isfinite(grad).all()
    assert all(p.grad is None for p in model.parameters())
    bias.zero_grad(); model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    loss2 = answer_losses(model, bias, [row], [layout], 0).mean()
    loss2.backward()
    torch.testing.assert_close(grad, bias.raw.grad)
    optimizer = torch.optim.Adam(bias.parameters(), lr=.1)
    initial = loss.item()
    for _ in range(12):
        optimizer.zero_grad()
        current = answer_losses(model, bias, [row], [layout], 0).mean()
        current.backward(); optimizer.step()
    assert current.item() < initial
    assert (bias.max_bias*bias.raw.tanh()).abs().max() <= bias.max_bias


def test_no_links_exact_baseline_and_causal_future_safety():
    model = tiny_model(); bias = StructureBias(2, 4, query_chunk=2); bias.attach(model)
    bias.set_layouts([{"pairs": [[0, 1, 4]]}])  # malicious future link must remain invisible
    tokens = torch.tensor([[1, 3, 4, 5, 6]])
    with torch.no_grad():
        a = model(tokens).logits
        bias.raw.fill_(1.)
        b = model(tokens).logits
    torch.testing.assert_close(a, b)
    bias.set_layouts([{"pairs": []}])
    with torch.no_grad(): c = model(tokens).logits
    torch.testing.assert_close(a, c)


def test_summary_uses_same_512_definitions_on_small_fixture():
    path = Path(__file__).resolve().parents[1] / "scripts/run_rag2_kg_attention_reporting_fix.py"
    spec = importlib.util.spec_from_file_location("kg_attention_cli", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = [{"sample_id": str(i), "dataset": d, "answer": "A"}
            for i, d in enumerate(("medmcqa", "medqa", "mmlu_anatomy"))]
    good = [{"sample_id": str(i), "prediction": "A", "finish_reason": "eos"} for i in range(3)]
    bad = [{**x, "prediction": None} for x in good]
    out = module.summarize(rows, {"no_rag": bad, "text": good, "graph_attention": good})
    assert out["conditions"]["text"]["macro3_pct"] == 100
    assert out["paired_comparisons"]["no_rag"]["delta_micro_pp"] == 100
    assert out["primary"]["transitions"]["C_C"] == 3
    assert not out["exploratory_gate"]["passed"]
    assert type(out["exploratory_gate"]["passed"]) is bool
    json.dumps(out, allow_nan=False)
    # The old fixture exercised a short-circuit false result only. A positive
    # delta evaluates the NumPy CI comparison and used to fail JSON encoding.
    improved = module.summarize(rows, {"no_rag": bad, "text": bad, "graph_attention": good})
    assert improved["exploratory_gate"]["passed"] is True
    json.dumps(improved, allow_nan=False)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(f) for name, f in globals().items()
                              if name.startswith("test_") and callable(f))


if __name__ == "__main__":
    unittest.main()
