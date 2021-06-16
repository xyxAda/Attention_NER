# coding: utf-8


import logging
from typing import Dict
from overrides import overrides
import torch


from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder,TimeDistributed,FeedForward
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules import FeedForward


from smhsan.modules.text_field_embedders import ShortcutConnectTextFieldEmbedder
from smhsan.models.type_extraction import TypeExtractor
from smhsan.models.layerBoundary import LayerBoundary
from allennlp.nn import RegularizerApplicator,InitializerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Model.register("smhsan")
class SMHSAN(Model):


    def __init__(self, vocab: Vocabulary,
                 params: Params,
                 regularizer: RegularizerApplicator = None):

        super(SMHSAN, self).__init__(vocab=vocab, regularizer=regularizer)

        # Base text Field Embedder
        text_field_embedder_params = params.pop("text_field_embedder")
        text_field_embedder = BasicTextFieldEmbedder.from_params(vocab=vocab, params=text_field_embedder_params)
        self._text_field_embedder = text_field_embedder

        ############
        # Boundary Detection Stuffs
        ############
        bd_params=params.pop('bd')
        encoder_bd_params = bd_params.pop("encoder")
        encoder_boundary = Seq2SeqEncoder.from_params(encoder_bd_params)

        self._encoder_bd = encoder_boundary
        mention_feedforward_params=bd_params.pop('tagger')
        self._mention_feedforward=FeedForward.from_params(mention_feedforward_params)
        initialiazer_params=bd_params.pop('initializer')
        self._initializer=InitializerApplicator.from_params(initialiazer_params)

        self._tagger_boundary_detection = LayerBoundary(vocab=vocab,
                                    text_field_embedder=self._text_field_embedder,
                                    dropout=bd_params.pop('dropout'),
                                    encoder_bd=self._encoder_bd,
                                    mention_feedforward=self._mention_feedforward)



        ############################
        # Entity Classification Stuffs
        ############################
        type_params = params.pop("ec")

        # Encoder
        encoder_type_params = type_params.pop("encoder")

        encoder_type = Seq2SeqEncoder.from_params(encoder_type_params)

        self._encoder_type = encoder_type

        shortcut_text_field_embedder_type = ShortcutConnectTextFieldEmbedder(
            base_text_field_embedder=self._text_field_embedder, previous_encoders=[self._encoder_bd]#, self._encoder_emd]
        )
        self._shortcut_text_field_embedder_type = shortcut_text_field_embedder_type


        tagger_type_params = type_params.pop("tagger")
        tagger_entity_classify = TypeExtractor(
            vocab=vocab,
            text_field_embedder=self._shortcut_text_field_embedder_type,
            context_layer=self._encoder_type,
            d=tagger_type_params.pop_int("d"),
            l=tagger_type_params.pop_int("l"),
            threshold=type_params.pop('threshold'),
            n_classes=tagger_type_params.pop("n_classes"),
            dropout=type_params.pop('dropout'),
            activation=tagger_type_params.pop("activation"),
        )

        self._tagger_entity_classify = tagger_entity_classify

        logger.info("Multi-Task Learning Model has been instantiated.")

    @overrides
    def forward(self, tensor_batch, for_training: bool = False, task_name: str = "boundary_detection") -> Dict[str, torch.Tensor]:
        tagger = getattr(self, "_tagger_%s" % task_name)
        return tagger.forward(**tensor_batch)

    @overrides
    def get_metrics(self, task_name: str, reset: bool = False, full: bool = False) -> Dict[str, float]:

        task_tagger = getattr(self, "_tagger_" + task_name)
        return task_tagger.get_metrics(reset)

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> "SMHSAN":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
