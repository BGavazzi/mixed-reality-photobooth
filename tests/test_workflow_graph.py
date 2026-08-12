"""
Tests for resolving workflow roles from graph structure.

The thing under test is a replacement for `POSITIVE_PROMPT_NODE = "7"`, so
the tests are mostly about the two ways that constant could be wrong: it can
point at nothing (KeyError mid-generation), or it can point at the wrong node
of the right kind -- writing the prompt into the negative conditioning, which
produces a plausible-looking bad image and reads as a model problem rather
than a bug. The second is the dangerous one, so most of what follows is about
never guessing.
"""

import pytest

import workflow_graph as wg
from workflow_graph import WorkflowSchemaError


def graph(**nodes):
    return dict(nodes)


def encode(text, clip="1"):
    return {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": [clip, 1]}}


SIMPLE = graph(
    n1={"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
    n2=encode("a prompt"),
    n3=encode("bad things"),
    n4={"class_type": "KSampler",
        "inputs": {"seed": 0, "denoise": 1.0, "positive": ["n2", 0], "negative": ["n3", 0]}},
)


# --- conditioning ---------------------------------------------------------------

def test_positive_and_negative_are_told_apart_by_where_they_are_wired():
    """Both are CLIPTextEncode nodes with identical shape. Only the link into
    the sampler distinguishes them -- which is exactly why a class-based or
    id-based lookup can get this silently wrong."""
    resolved = wg.resolve(SIMPLE)
    assert resolved.node_id(wg.POSITIVE_PROMPT) == "n2"
    assert resolved.node_id(wg.NEGATIVE_PROMPT) == "n3"


def test_swapping_the_links_swaps_the_roles():
    """The strongest form of the above: same nodes, same ids, same order --
    only the wiring changes, and the answer follows the wiring."""
    swapped = {k: dict(v) for k, v in SIMPLE.items()}
    swapped["n4"] = {"class_type": "KSampler",
                     "inputs": {"seed": 0, "positive": ["n3", 0], "negative": ["n2", 0]}}
    resolved = wg.resolve(swapped)
    assert resolved.node_id(wg.POSITIVE_PROMPT) == "n3"
    assert resolved.node_id(wg.NEGATIVE_PROMPT) == "n2"


def test_conditioning_is_traced_through_pass_through_nodes():
    """The real photoshoot graph puts ControlNetApplyAdvanced between the text
    encoders and the sampler, so a one-hop lookup finds the ControlNet node
    rather than the prompt."""
    chained = graph(
        n2=encode("a prompt"),
        n3=encode("bad things"),
        n9={"class_type": "ControlNetApplyAdvanced",
            "inputs": {"positive": ["n2", 0], "negative": ["n3", 0],
                       "control_net": ["n8", 0], "image": ["n7", 0], "strength": 0.75}},
        n7={"class_type": "LoadImage", "inputs": {"image": "depth.png"}},
        n8={"class_type": "ControlNetLoader", "inputs": {"control_net_name": "d.safetensors"}},
        n4={"class_type": "KSampler",
            "inputs": {"seed": 0, "positive": ["n9", 0], "negative": ["n9", 1]}},
    )
    resolved = wg.resolve(chained)
    assert resolved.node_id(wg.POSITIVE_PROMPT) == "n2"
    assert resolved.node_id(wg.DEPTH_IMAGE) == "n7"


def test_an_unknown_node_in_the_chain_stops_the_walk_rather_than_guessing():
    """A wrong answer is worse than no answer here. An unrecognised
    conditioning node should leave the role unresolved -- which then fails
    loudly via require= -- not cause a sideways search for any text encoder."""
    exotic = graph(
        n2=encode("a prompt"),
        n5={"class_type": "SomeCustomConditioningNode", "inputs": {"positive": ["n2", 0]}},
        n4={"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["n5", 0]}},
    )
    resolved = wg.resolve(exotic)
    assert not resolved.has(wg.POSITIVE_PROMPT)

    with pytest.raises(WorkflowSchemaError):
        wg.resolve(exotic, require=(wg.POSITIVE_PROMPT,))


def test_a_cyclic_graph_terminates():
    """Malformed input shouldn't hang or blow the stack."""
    cyclic = graph(
        a={"class_type": "ConditioningCombine", "inputs": {"positive": ["b", 0]}},
        b={"class_type": "ConditioningCombine", "inputs": {"positive": ["a", 0]}},
        k={"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["a", 0]}},
    )
    assert not wg.resolve(cyclic).has(wg.POSITIVE_PROMPT)


# --- samplers -------------------------------------------------------------------

def test_no_sampler_is_refused_immediately():
    with pytest.raises(WorkflowSchemaError, match="no sampler"):
        wg.resolve(graph(n1=encode("x")))


def test_two_samplers_are_refused_rather_than_guessed():
    """A refiner-pass graph has a real answer, but it isn't derivable from
    class alone -- and picking the first would be arbitrary and silent."""
    two = dict(SIMPLE)
    two["n5"] = {"class_type": "KSampler", "inputs": {"seed": 0}}
    with pytest.raises(WorkflowSchemaError, match="sampler nodes"):
        wg.resolve(two)


@pytest.mark.parametrize("sampler_class,field", [
    ("KSampler", "seed"),
    ("SamplerCustom", "noise_seed"),
    ("KSamplerAdvanced", "noise_seed"),
])
def test_the_seed_field_follows_the_sampler_class(sampler_class, field):
    """SamplerCustom names it `noise_seed`. That difference used to live as a
    comment beside a hardcoded node id in a different file."""
    workflow = graph(k={"class_type": sampler_class, "inputs": {field: 0}})
    resolved = wg.resolve(workflow)
    assert resolved.seed_field == field
    resolved.set_seed(4242)
    assert workflow["k"]["inputs"][field] == 4242


# --- images ---------------------------------------------------------------------

def test_the_three_image_loaders_are_told_apart_by_their_consumers():
    """The photoshoot graph loads three images through near-identical nodes.
    Class alone cannot say which is the subject -- the consumer can."""
    workflow = graph(
        subj={"class_type": "LoadImage", "inputs": {"image": "s.png"}},
        msk={"class_type": "LoadImageMask", "inputs": {"image": "m.png", "channel": "red"}},
        dep={"class_type": "LoadImage", "inputs": {"image": "d.png"}},
        enc={"class_type": "VAEEncode", "inputs": {"pixels": ["subj", 0], "vae": ["ck", 2]}},
        lat={"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["enc", 0], "mask": ["msk", 0]}},
        cn={"class_type": "ControlNetApplyAdvanced",
            "inputs": {"image": ["dep", 0], "positive": ["p", 0], "negative": ["n", 0], "strength": 0.5}},
        p=encode("yes"), n=encode("no"),
        k={"class_type": "KSampler", "inputs": {"seed": 0, "positive": ["cn", 0], "negative": ["cn", 1]}},
    )
    resolved = wg.resolve(workflow)
    assert resolved.node_id(wg.SUBJECT_IMAGE) == "subj"
    assert resolved.node_id(wg.MASK_IMAGE) == "msk"
    assert resolved.node_id(wg.DEPTH_IMAGE) == "dep"


def test_a_literal_input_is_not_mistaken_for_a_link():
    """`["4", 0]` is a link; a literal filename is not. Getting this wrong
    would have the resolver chasing strings as node ids."""
    assert wg._is_link(["4", 0])
    assert not wg._is_link("a_photo.png")
    assert not wg._is_link([0.5, 0.5])
    assert not wg._is_link(["4", 0, 1])


# --- patching --------------------------------------------------------------------

def test_set_writes_through_the_role():
    workflow = {k: dict(v) for k, v in SIMPLE.items()}
    workflow["n2"] = dict(workflow["n2"], inputs=dict(workflow["n2"]["inputs"]))
    resolved = wg.resolve(workflow)
    resolved.set(wg.POSITIVE_PROMPT, "text", "something else")
    assert workflow["n2"]["inputs"]["text"] == "something else"


def test_setting_a_field_the_node_does_not_have_is_refused():
    """The other half of drift: a node of the right class whose inputs changed
    shape between ComfyUI versions would otherwise silently gain a key the
    sampler never reads."""
    resolved = wg.resolve(SIMPLE)
    with pytest.raises(WorkflowSchemaError, match="no input"):
        resolved.set(wg.POSITIVE_PROMPT, "prompt", "wrong field name")


def test_set_if_present_is_a_no_op_for_an_absent_role():
    """txt2img has no ControlNet, and callers shouldn't have to branch."""
    wg.resolve(SIMPLE).set_if_present(wg.CONTROLNET, "strength", 0.5)


def test_asking_for_an_unresolved_role_names_what_was_resolved():
    resolved = wg.resolve(SIMPLE)
    with pytest.raises(WorkflowSchemaError) as excinfo:
        resolved.node_id(wg.DEPTH_IMAGE)
    assert wg.POSITIVE_PROMPT in str(excinfo.value)


# --- loading ---------------------------------------------------------------------

def test_a_ui_format_workflow_is_rejected_with_the_reason(tmp_path):
    """`Save` and `Save (API format)` produce different shapes and only the
    second is executable. Exporting the wrong one is a common first mistake,
    and 'list indices must be integers' would not explain it."""
    path = tmp_path / "ui_format.json"
    path.write_text('{"nodes": [{"id": 1}], "links": []}', encoding="utf-8")
    with pytest.raises(WorkflowSchemaError, match="API format"):
        wg.load(path)


def test_model_names_reads_both_loaders():
    workflow = graph(
        a={"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "m.safetensors"}},
        b={"class_type": "ControlNetLoader", "inputs": {"control_net_name": "d.safetensors"}},
    )
    assert wg.model_names(workflow) == {"checkpoint": "m.safetensors", "controlnet": "d.safetensors"}
