from transformers import AutoModelForQuestionAnswering
from transformers.models.auto.auto_factory import _LazyAutoMapping
from collections import OrderedDict
from configuration_bert import BertConfigDFSS
from modeling_bert import BertForQuestionAnsweringDFSS
import transformers

from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.auto_factory import _get_model_class

setattr(transformers, "BertForQuestionAnsweringDFSS", BertForQuestionAnsweringDFSS)
setattr(transformers, "BertConfigDFSS", BertConfigDFSS)


DFSS_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # DFSS Models for Question Answering mapping
        ("bert", "BertForQuestionAnsweringDFSS"),
    ]
)

DFSS_CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("bert", "BertConfigDFSS")
    ]
)


DFSS_MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    DFSS_CONFIG_MAPPING_NAMES, DFSS_MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)

class DFSSAutoModelForQuestionAnswering(AutoModelForQuestionAnswering):
    _model_mapping = DFSS_MODEL_FOR_QUESTION_ANSWERING_MAPPING

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        kwargs["_from_auto"] = True
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in cls._model_mapping.keys():
            model_class = _get_model_class(config, cls._model_mapping)
            return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoModel: {cls.__name__}.\n"
            f"Model type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )