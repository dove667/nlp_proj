from __future__ import annotations

from models.base import GenerationModel, InferenceMethod
from models.llama.model import LlamaSourceModel
from models.longformer.model import LongformerSourceModel
from models.mamba2.model import Mamba2SourceModel
from models.jamba.model import JambaSourceModel
from models.methods.baseline import BaselineMethod
from models.methods.fier import FIERMethod
from models.methods.kivi import KIVIMethod
from models.methods.self_extend import SelfExtendMethod
from models.methods.snapkv import SnapKVMethod
from models.methods.streaming_llm import StreamingLLMMethod
from models.methods.yarn import YaRNMethod
from models.spec import MethodSpec, ModelSpec


def build_model(spec: ModelSpec) -> GenerationModel:
    implementations = {
        "llama": LlamaSourceModel,
        "longformer": LongformerSourceModel,
        "mamba2": Mamba2SourceModel,
        "jamba": JambaSourceModel,
    }
    try:
        cls = implementations[spec.implementation]
    except KeyError as exc:
        raise ValueError(f"Unknown model implementation: {spec.implementation}") from exc
    return cls(spec)


def build_method(base_model: GenerationModel, spec: MethodSpec) -> InferenceMethod:
    implementations = {
        "baseline": BaselineMethod,
        "yarn": YaRNMethod,
        "self_extend": SelfExtendMethod,
        "kivi": KIVIMethod,
        "snapkv": SnapKVMethod,
        "fier": FIERMethod,
        "streaming_llm": StreamingLLMMethod,
    }
    try:
        cls = implementations[spec.implementation]
    except KeyError as exc:
        raise ValueError(f"Unknown method implementation: {spec.implementation}") from exc
    return cls(base_model, spec)
