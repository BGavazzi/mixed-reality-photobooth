"""
Tests that the AI-disclosure provenance record names the model that
actually ran.

These names used to be hardcoded constants duplicating the workflow JSON, in
two separate places (the server and the browser's project export). Swapping
the checkpoint inside the workflow left both silently reporting the old
name -- a quiet wrong answer on the one panel in the UI whose entire purpose
is being accurate about what generated an image.
"""

import json

import pytest

from backends.comfy import DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH
from web_server import _read_model_names_from_workflow


def write_workflow(tmp_path, nodes):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(nodes), encoding="utf-8")
    return path


def test_reads_both_model_names_from_the_real_workflow():
    checkpoint, controlnet = _read_model_names_from_workflow(DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH)
    assert checkpoint and checkpoint.endswith(".safetensors")
    assert controlnet and controlnet.endswith(".safetensors")


def test_lookup_is_by_node_class_not_node_id(tmp_path):
    """The whole point of matching on class_type: re-exporting a workflow
    from the ComfyUI UI renumbers its nodes, and an ID-based lookup would
    then read the wrong node -- or none at all -- without failing."""
    renumbered = write_workflow(tmp_path, {
        "873": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "someModel.safetensors"}},
        "12": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "someDepth.safetensors"}},
        "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
    })
    assert _read_model_names_from_workflow(renumbered) == ("someModel.safetensors", "someDepth.safetensors")


def test_a_swapped_checkpoint_is_reflected_immediately(tmp_path):
    for name in ("realvis.safetensors", "juggernaut.safetensors"):
        path = write_workflow(tmp_path, {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}},
            "2": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "depth.safetensors"}},
        })
        assert _read_model_names_from_workflow(path)[0] == name


def test_missing_loader_nodes_warn_rather_than_crash(tmp_path, capsys):
    """A degraded provenance field must not take the whole module import
    down -- but it must not pass unmentioned either."""
    path = write_workflow(tmp_path, {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}})

    assert _read_model_names_from_workflow(path) == (None, None)
    assert "warning" in capsys.readouterr().out.lower()


def test_workflow_without_a_controlnet_still_reports_its_checkpoint(tmp_path):
    """The plain txt2img workflow has no ControlNet at all; the checkpoint
    name is still worth recording."""
    path = write_workflow(tmp_path, {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
    })
    assert _read_model_names_from_workflow(path) == ("sdxl.safetensors", None)


@pytest.mark.parametrize("workflow_name", ["photoshoot_bg_api.json", "txt2img_api.json", "text_to_video_api.json"])
def test_shipped_workflows_are_valid_json_with_typed_nodes(workflow_name):
    """Cheap guard against a truncated or hand-edited workflow file: every
    node the app indexes into must at least be a dict with a class_type."""
    path = DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH.parent / workflow_name
    workflow = json.loads(path.read_text(encoding="utf-8"))
    assert workflow, f"{workflow_name} is empty"
    for node_id, node in workflow.items():
        assert isinstance(node, dict), f"node {node_id} in {workflow_name} is not an object"
        assert node.get("class_type"), f"node {node_id} in {workflow_name} has no class_type"


def test_node_ids_the_backend_hardcodes_still_exist():
    """backends/comfy.py addresses workflow nodes by hardcoded ID (a known
    limitation in the README). If a workflow is ever re-exported and
    renumbered, this fails here instead of at generation time with a
    KeyError -- or, worse, by writing a prompt into the wrong node.
    """
    from backends import comfy

    checks = [
        (comfy.DEFAULT_WORKFLOW_PATH, {
            comfy.POSITIVE_PROMPT_NODE: "text",
            comfy.SEED_NODE: "seed",
        }),
        (comfy.DEFAULT_VIDEO_WORKFLOW_PATH, {
            comfy.VIDEO_POSITIVE_PROMPT_NODE: "text",
            comfy.VIDEO_SEED_NODE: "noise_seed",
        }),
        (comfy.DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH, {
            comfy.PHOTOSHOOT_SUBJECT_NODE: "image",
            comfy.PHOTOSHOOT_MASK_NODE: "image",
            comfy.PHOTOSHOOT_DEPTH_NODE: "image",
            comfy.PHOTOSHOOT_POSITIVE_PROMPT_NODE: "text",
            comfy.PHOTOSHOOT_CONTROLNET_NODE: "strength",
            comfy.PHOTOSHOOT_SEED_NODE: "seed",
        }),
    ]
    for path, expected in checks:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        for node_id, input_key in expected.items():
            assert node_id in workflow, f"{path.name} has no node {node_id}"
            assert input_key in workflow[node_id]["inputs"], \
                f"{path.name} node {node_id} has no '{input_key}' input"
