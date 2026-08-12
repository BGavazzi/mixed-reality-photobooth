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

import workflow_graph
from backends.comfy import DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH
from web_server import _read_workflow_facts


def model_names(path):
    """The two model fields, dropping the negative prompt this file doesn't
    test (see test_brand_kit.py for that half)."""
    checkpoint, controlnet, _negative = _read_workflow_facts(path)
    return checkpoint, controlnet


def write_workflow(tmp_path, nodes):
    path = tmp_path / "workflow.json"
    path.write_text(json.dumps(nodes), encoding="utf-8")
    return path


def test_reads_both_model_names_from_the_real_workflow():
    checkpoint, controlnet = model_names(DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH)
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
    assert model_names(renumbered) == ("someModel.safetensors", "someDepth.safetensors")


def test_a_swapped_checkpoint_is_reflected_immediately(tmp_path):
    for name in ("realvis.safetensors", "juggernaut.safetensors"):
        path = write_workflow(tmp_path, {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}},
            "2": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "depth.safetensors"}},
        })
        assert model_names(path)[0] == name


def test_missing_loader_nodes_warn_rather_than_crash(tmp_path, capsys):
    """A degraded provenance field must not take the whole module import
    down -- but it must not pass unmentioned either."""
    path = write_workflow(tmp_path, {"1": {"class_type": "KSampler", "inputs": {"seed": 1}}})

    assert model_names(path) == (None, None)
    assert "warning" in capsys.readouterr().out.lower()


def test_workflow_without_a_controlnet_still_reports_its_checkpoint(tmp_path):
    """The plain txt2img workflow has no ControlNet at all; the checkpoint
    name is still worth recording."""
    path = write_workflow(tmp_path, {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sdxl.safetensors"}},
    })
    assert model_names(path) == ("sdxl.safetensors", None)


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


def test_every_shipped_workflow_resolves_the_roles_its_entry_point_needs():
    """This replaces a test that asserted a set of hardcoded node ids still
    existed in each workflow. That test guarded a limitation rather than a
    behaviour -- it could only ever tell you that a re-export had already
    broken the backend. Now that roles are resolved from each graph's own
    wiring (workflow_graph.py), the useful question is whether the shipped
    workflows still *have* the structure each entry point requires.
    """
    from backends import comfy

    for path, required in (
        (comfy.DEFAULT_WORKFLOW_PATH, comfy.TXT2IMG_ROLES),
        (comfy.DEFAULT_VIDEO_WORKFLOW_PATH, comfy.VIDEO_ROLES),
        (comfy.DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH, comfy.PHOTOSHOOT_ROLES),
    ):
        resolved = workflow_graph.load(path, require=required)
        for role in required:
            assert resolved.has(role), f"{path.name} lost role {role}"


def test_the_resolver_survives_a_wholesale_renumbering():
    """The property the old hardcoded-id test could not check at all: shuffle
    every node id in the real photoshoot workflow and the same roles must
    still land on the same nodes."""
    workflow = json.loads(DEFAULT_PHOTOSHOOT_BG_WORKFLOW_PATH.read_text(encoding="utf-8"))
    remap = {old: f"{9000 + i}" for i, old in enumerate(workflow)}

    renumbered = {}
    for old_id, node in workflow.items():
        clone = json.loads(json.dumps(node))
        for key, value in clone.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and value[0] in remap:
                clone["inputs"][key] = [remap[value[0]], value[1]]
        renumbered[remap[old_id]] = clone

    before = workflow_graph.resolve(workflow, require=comfy_photoshoot_roles())
    after = workflow_graph.resolve(renumbered, require=comfy_photoshoot_roles())

    assert {role: remap[nid] for role, nid in before.roles.items()} == after.roles


def comfy_photoshoot_roles():
    from backends import comfy
    return comfy.PHOTOSHOOT_ROLES


def test_a_workflow_missing_a_required_role_fails_at_load_with_a_useful_message():
    """The failure mode this whole module exists to improve: previously a
    missing node surfaced as a KeyError mid-generation, after the user had
    already waited. Now it names the file, the role, and what was found."""
    from backends import comfy

    no_controlnet = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "x.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "hi", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "no", "clip": ["1", 1]}},
        "4": {"class_type": "KSampler",
              "inputs": {"seed": 1, "positive": ["2", 0], "negative": ["3", 0]}},
    }
    with pytest.raises(workflow_graph.WorkflowSchemaError) as excinfo:
        workflow_graph.resolve(no_controlnet, source="fake.json", require=comfy.PHOTOSHOOT_ROLES)

    message = str(excinfo.value)
    assert "fake.json" in message
    assert workflow_graph.CONTROLNET in message
    assert "KSampler" in message, "the message should show what the graph does contain"
