"""
Resolving workflow node IDs by what a node *does*, not by what number it got.

Every workflow input this app patches used to be addressed by a hardcoded
numeric ID:

    POSITIVE_PROMPT_NODE = "7"
    workflow["7"]["inputs"]["text"] = prompt

Those numbers are an artifact of one export from ComfyUI's UI. Re-export the
same graph after moving a node, or rebuild it from scratch, and the numbering
changes -- at which point the app either patches the *wrong* node (silently
sending the prompt into the negative conditioning, which looks like a model
quality problem, not a bug) or raises KeyError deep inside a generation the
user is already waiting on. Neither failure mentions the workflow file.

So instead of trusting the numbering, this module reads the graph. A
workflow is a set of nodes plus their links, which is enough to answer "which
node is the positive prompt" structurally: it is the CLIPTextEncode that
reaches the sampler's `positive` input. That is the same question ComfyUI
itself answers at execution time, and the answer does not depend on any node
ID being stable.

Two consequences worth naming, since they are the point of the exercise:

  * Re-exporting a workflow no longer breaks the backend.
  * When a workflow genuinely *doesn't* have what the code needs -- someone
    deleted the ControlNet, or wired the sampler straight to a checkpoint's
    CLIP -- it fails at load with a message naming the file, the role, and
    what was looked for, instead of at generation time with a KeyError.

The tradeoff, stated plainly: this is more code than a dict of constants, and
it assumes the graph shapes this app ships. A workflow that puts something
exotic between the text encoder and the sampler will not resolve, and should
be added to PASS_THROUGH_CONDITIONING rather than worked around at the call
site.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# --- roles -----------------------------------------------------------------
# The semantic names the rest of the app uses. Everything outside this module
# refers to these, never to a node id.

POSITIVE_PROMPT = "positive_prompt"
NEGATIVE_PROMPT = "negative_prompt"
SUBJECT_IMAGE = "subject_image"
MASK_IMAGE = "mask_image"
DEPTH_IMAGE = "depth_image"
CONTROLNET = "controlnet"
SAMPLER = "sampler"

# Sampler classes this app knows how to drive, mapped to the input that holds
# their noise seed. SamplerCustom calls it `noise_seed` rather than `seed`,
# which was previously a second hardcoded fact (VIDEO_SEED_NODE's comment) in
# a different file from the node id it belonged to.
SAMPLER_SEED_FIELDS = {
    "KSampler": "seed",
    "KSamplerAdvanced": "noise_seed",
    "SamplerCustom": "noise_seed",
    "SamplerCustomAdvanced": "noise_seed",
}

# Nodes that transform conditioning without being its origin. Walking back
# from a sampler's `positive` input passes through these to reach the text
# encoder that actually holds the prompt.
PASS_THROUGH_CONDITIONING = {
    "ControlNetApplyAdvanced",
    "ControlNetApply",
    "ConditioningCombine",
    "ConditioningConcat",
    "ConditioningSetArea",
    "ConditioningZeroOut",
    # LTXV (the text-to-video path) stamps a frame rate onto both conditioning
    # branches on the way to the sampler. Its `positive`/`negative` inputs are
    # named identically to the sampler's, which is what makes the same
    # same-key backward walk work across an image and a video graph.
    "LTXVConditioning",
}

TEXT_ENCODER_CLASSES = {"CLIPTextEncode", "CLIPTextEncodeSDXL"}
IMAGE_LOADER_CLASSES = {"LoadImage", "LoadImageMask", "LoadImageOutput"}

# Depth of the backward walk. Generous enough for any conditioning chain this
# app would plausibly meet, bounded so a cyclic or self-referential graph
# fails with a message instead of recursing until the interpreter gives up.
MAX_WALK_DEPTH = 12


class WorkflowSchemaError(ValueError):
    """A workflow does not contain a node the code needs.

    Raised at load time on purpose. The alternative -- discovering it when a
    KeyError surfaces mid-generation -- costs the user their wait and tells
    them nothing about which file is wrong.
    """


# --- graph primitives --------------------------------------------------------

def _is_link(value) -> bool:
    """ComfyUI encodes a link as [node_id, output_index]; a literal is
    anything else. `["4", 0]` is a link, `"a prompt"` and `0.75` are not."""
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


def _linked_node(workflow: dict, node_id: str, input_key: str) -> str | None:
    """The node id feeding `input_key` of `node_id`, or None if that input is
    a literal, absent, or points at a node that isn't in the graph."""
    node = workflow.get(node_id)
    if not node:
        return None
    value = node.get("inputs", {}).get(input_key)
    if not _is_link(value):
        return None
    return value[0] if value[0] in workflow else None


def _class_of(workflow: dict, node_id: str) -> str:
    return workflow.get(node_id, {}).get("class_type", "")


def find_by_class(workflow: dict, *class_types: str) -> list[str]:
    """Every node id of the given class(es), in workflow order."""
    wanted = set(class_types)
    return [nid for nid, node in workflow.items() if node.get("class_type") in wanted]


def _walk_back(workflow: dict, start_id: str, input_key: str,
               targets: set[str], through: set[str]) -> str | None:
    """Follows `input_key` backwards from `start_id` until it reaches a node
    whose class is in `targets`, passing only through classes in `through`.

    Stops at anything unexpected rather than searching sideways: a wrong
    answer here is worse than no answer, because it would send a prompt into
    the wrong conditioning slot and present as a model-quality problem.
    """
    current = _linked_node(workflow, start_id, input_key)
    for _ in range(MAX_WALK_DEPTH):
        if current is None:
            return None
        node_class = _class_of(workflow, current)
        if node_class in targets:
            return current
        if node_class not in through:
            return None
        current = _linked_node(workflow, current, input_key)
    return None


def _trace_image_source(workflow: dict, consumer_id: str, input_key: str) -> str | None:
    """The image-loading node feeding `input_key` of `consumer_id`.

    Unlike conditioning there is no chain to walk here in the workflows this
    app ships -- the loaders wire straight into their consumer -- so this is
    one hop plus a class check, which keeps a wrong answer impossible rather
    than unlikely.
    """
    source = _linked_node(workflow, consumer_id, input_key)
    if source and _class_of(workflow, source) in IMAGE_LOADER_CLASSES:
        return source
    return None


# --- the resolved object the rest of the app uses -----------------------------

@dataclass
class ResolvedWorkflow:
    """A workflow plus the node ids its semantic roles landed on.

    Holds the workflow dict itself so callers patch through `set()` and never
    need a node id at all. `source` is carried purely so an error names the
    file a human has to go and fix.
    """
    workflow: dict
    roles: dict[str, str]
    source: Path | None = None

    def node_id(self, role: str) -> str:
        try:
            return self.roles[role]
        except KeyError:
            raise WorkflowSchemaError(
                f"{self.source or 'workflow'}: no node resolved for role {role!r} "
                f"(resolved roles: {sorted(self.roles)})") from None

    def has(self, role: str) -> bool:
        return role in self.roles

    def set(self, role: str, field: str, value):
        """Patches one input of the node holding `role`.

        Validates that the field already exists, which catches the other half
        of the drift problem: a node of the right class whose inputs changed
        shape between ComfyUI versions would otherwise silently gain a new
        key that the sampler never reads.
        """
        node_id = self.node_id(role)
        inputs = self.workflow[node_id].setdefault("inputs", {})
        if field not in inputs:
            raise WorkflowSchemaError(
                f"{self.source or 'workflow'}: node {node_id} "
                f"({_class_of(self.workflow, node_id)}) fills role {role!r} but has no "
                f"input {field!r} to set (has: {sorted(inputs)})")
        inputs[field] = value

    def set_if_present(self, role: str, field: str, value):
        """For roles a workflow may legitimately lack -- txt2img has no
        ControlNet, and a caller shouldn't have to branch on that."""
        if self.has(role):
            self.set(role, field, value)

    @property
    def seed_field(self) -> str:
        """`seed` or `noise_seed`, per the sampler's own class."""
        sampler_class = _class_of(self.workflow, self.node_id(SAMPLER))
        return SAMPLER_SEED_FIELDS[sampler_class]

    def set_seed(self, seed: int):
        self.set(SAMPLER, self.seed_field, seed)

    def describe(self) -> str:
        """Human-readable role -> node mapping, for diagnostics."""
        return "\n".join(
            f"  {role:<16} -> node {nid} ({_class_of(self.workflow, nid)})"
            for role, nid in sorted(self.roles.items()))


# --- resolution ---------------------------------------------------------------

def _resolve_sampler(workflow: dict, source) -> str:
    samplers = find_by_class(workflow, *SAMPLER_SEED_FIELDS)
    if not samplers:
        raise WorkflowSchemaError(
            f"{source}: no sampler node found (looked for any of "
            f"{sorted(SAMPLER_SEED_FIELDS)}) -- this app has to set a seed, "
            f"so a workflow without one cannot be driven")
    if len(samplers) > 1:
        # Deliberately refused rather than guessed. A two-sampler graph (a
        # refiner pass, say) has a real answer, but it isn't derivable from
        # class alone and picking the first would be arbitrary.
        raise WorkflowSchemaError(
            f"{source}: {len(samplers)} sampler nodes ({', '.join(samplers)}); "
            f"this app drives single-sampler workflows and won't guess which "
            f"one owns the seed")
    return samplers[0]


def resolve(workflow: dict, source=None, require: tuple[str, ...] = ()) -> ResolvedWorkflow:
    """Maps a workflow's nodes onto the roles this app patches.

    `require` names the roles the caller cannot work without, so each entry
    point declares its own needs -- txt2img legitimately has no ControlNet or
    mask, and shouldn't be held to the photoshoot workflow's shape.
    """
    source = source or "workflow"
    roles: dict[str, str] = {}

    sampler_id = _resolve_sampler(workflow, source)
    roles[SAMPLER] = sampler_id

    # Conditioning: whichever text encoder reaches the sampler's positive and
    # negative inputs, through any number of ControlNet/conditioning nodes.
    for role, input_key in ((POSITIVE_PROMPT, "positive"), (NEGATIVE_PROMPT, "negative")):
        found = _walk_back(workflow, sampler_id, input_key,
                           targets=TEXT_ENCODER_CLASSES, through=PASS_THROUGH_CONDITIONING)
        if found:
            roles[role] = found

    controlnets = find_by_class(workflow, "ControlNetApplyAdvanced", "ControlNetApply")
    if len(controlnets) == 1:
        roles[CONTROLNET] = controlnets[0]
        depth = _trace_image_source(workflow, controlnets[0], "image")
        if depth:
            roles[DEPTH_IMAGE] = depth

    # The init image is whatever gets VAE-encoded into the sampler's latent;
    # the inpaint mask is whatever SetLatentNoiseMask is given. Both are found
    # through their consumer rather than by loader class, because LoadImage
    # appears three times in the photoshoot graph and the class alone cannot
    # say which is which.
    for encoder_id in find_by_class(workflow, "VAEEncode", "VAEEncodeForInpaint"):
        subject = _trace_image_source(workflow, encoder_id, "pixels")
        if subject:
            roles[SUBJECT_IMAGE] = subject
            break

    for mask_node_id in find_by_class(workflow, "SetLatentNoiseMask"):
        mask = _trace_image_source(workflow, mask_node_id, "mask")
        if mask:
            roles[MASK_IMAGE] = mask
            break

    missing = [role for role in require if role not in roles]
    if missing:
        raise WorkflowSchemaError(
            f"{source}: could not resolve required role(s) {missing} from this graph.\n"
            f"What was resolved:\n{ResolvedWorkflow(workflow, roles, source).describe()}\n"
            f"Node classes present: {sorted({n.get('class_type', '?') for n in workflow.values()})}")

    return ResolvedWorkflow(workflow, roles, source)


def load(path: Path, require: tuple[str, ...] = ()) -> ResolvedWorkflow:
    """Reads a workflow file and resolves it. Each call gets its own copy of
    the dict, since callers patch it per generation."""
    path = Path(path)
    with open(path) as f:
        workflow = json.load(f)
    _require_api_format(workflow, path)
    return resolve(workflow, source=path, require=require)


def _require_api_format(workflow, source) -> None:
    """Rejects a workflow exported with plain `Save` instead of
    `Save (API format)`.

    Worth its own check with its own message: exporting the wrong one is a
    common first mistake, the two files look equally plausible sitting in a
    folder, and neither ComfyUI's /prompt nor a naive parse says anything
    useful about it. Note that a UI-format export is *also* a JSON object, so
    an isinstance(dict) test passes it straight through -- it just maps
    "nodes"/"links"/"groups" instead of node ids.
    """
    if not isinstance(workflow, dict):
        raise WorkflowSchemaError(
            f"{source}: expected an API-format workflow object (node id -> node), "
            f"got {type(workflow).__name__}. Re-export with 'Save (API format)'.")
    if isinstance(workflow.get("nodes"), list):
        raise WorkflowSchemaError(
            f"{source}: this looks like a UI-format workflow (it has a top-level "
            f"'nodes' list). Only the API format -- ComfyUI's 'Save (API format)', "
            f"which maps node id -> node -- is executable by /prompt.")
    bad = [k for k, v in workflow.items() if not isinstance(v, dict)]
    if bad:
        raise WorkflowSchemaError(
            f"{source}: entries {bad[:3]} are not node objects; this is not an "
            f"API-format workflow. Re-export with 'Save (API format)'.")


def model_names(workflow: dict) -> dict[str, str | None]:
    """Checkpoint and ControlNet filenames, by loader class.

    Lives here rather than in web_server.py so there is one place that knows
    how to read a fact out of a workflow.
    """
    names: dict[str, str | None] = {"checkpoint": None, "controlnet": None}
    for node in workflow.values():
        class_type = node.get("class_type")
        if class_type == "CheckpointLoaderSimple":
            names["checkpoint"] = node.get("inputs", {}).get("ckpt_name")
        elif class_type == "ControlNetLoader":
            names["controlnet"] = node.get("inputs", {}).get("control_net_name")
    return names


def main():
    """`python workflow_graph.py` prints the role map for every shipped
    workflow -- the fastest way to check a re-export before it reaches a
    generation."""
    import sys
    workflows = sorted((Path(__file__).parent / "workflows").glob("*.json"))
    if not workflows:
        print("no workflows found")
        return 1
    failed = False
    for path in workflows:
        print(f"\n{path.name}")
        try:
            resolved = load(path)
        except (WorkflowSchemaError, json.JSONDecodeError) as exc:
            print(f"  FAILED: {exc}")
            failed = True
            continue
        print(resolved.describe())
        print(f"  {'seed field':<16} -> {resolved.seed_field}")
        names = model_names(resolved.workflow)
        print(f"  {'checkpoint':<16} -> {names['checkpoint']}")
        print(f"  {'controlnet':<16} -> {names['controlnet']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
