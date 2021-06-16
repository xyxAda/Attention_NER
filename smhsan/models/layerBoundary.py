# coding: utf-8


from typing import Dict, Optional, List, Any
import logging
from overrides import overrides

import torch
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder,FeedForward,TimeDistributed
from allennlp.nn import RegularizerApplicator,InitializerApplicator
from smhsan.training.metrics import BDMetrics
from allennlp.nn import util
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def replace_masked_values(tensor: torch.Tensor, mask: torch.Tensor, replace_with: float) -> torch.Tensor:
    """
    Replaces all masked values in ``tensor`` with ``replace_with``.  ``mask`` must be broadcastable
    to the same shape as ``tensor``. We require that ``tensor.dim() == mask.dim()``, as otherwise we
    won't know which dimensions of the mask to unsqueeze.

    This just does ``tensor.masked_fill()``, except the pytorch method fills in things with a mask
    value of 1, where we want the opposite.  You can do this in your own code with
    ``tensor.masked_fill((1 - mask).to(dtype=torch.bool), replace_with)``.
    """
    if tensor.dim() != mask.dim():
        raise ConfigurationError("tensor.dim() (%d) != mask.dim() (%d)" % (tensor.dim(), mask.dim()))
    return tensor.masked_fill(torch.tensor((1 - mask),dtype=torch.uint8).cuda(tensor.device), replace_with)


@Model.register("bd")
class LayerBoundary(Model):
    """
    A class that implement the first task of HMTL model: NER (CRF Tagger).
    
    Parameters
    ----------
    vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
    params: ``allennlp.common.Params``, required
        Configuration parameters for the multi-task model.
    regularizer: ``allennlp.nn.RegularizerApplicator``, optional (default = None)
        A reguralizer to apply to the model's layers.
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder_bd:Seq2SeqEncoder,
                 mention_feedforward:FeedForward,
                 dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: RegularizerApplicator = None):

        super(LayerBoundary, self).__init__(vocab=vocab, regularizer=regularizer)


        ############
        # Boundary Detection Stuffs
        ############
        self._n_labels = vocab.get_vocab_size('json_bd_labels')

        # TODO(dwadden) think of a better way to enforce this.
        # Null label is needed to keep track of when calculating the metrics
        null_label = vocab.get_token_index("O", "json_bd_labels")
        assert null_label == 0

        self._text_field_embedder = text_field_embedder
        self._encoder_bd = encoder_bd
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None


        # Tagger Boundary Detection
        self._bd_scorer = torch.nn.Sequential(
            TimeDistributed(mention_feedforward),
            TimeDistributed(torch.nn.Linear(
                mention_feedforward.get_output_dim(),
                self._n_labels - 1)))


        self._bd_metrics = BDMetrics(self._n_labels, null_label)

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum")





    @overrides
    def forward(self,  # type: ignore
                tokens: Dict[str, torch.LongTensor],
                tags: torch.LongTensor = None,
                **kwargs) -> Dict[str, torch.Tensor]:

        embedded_text_input = self._text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self._encoder_bd(embedded_text_input, mask)

        bd_scores = self._bd_scorer(encoded_text)
        mask_fi=mask.unsqueeze(-1)
        bd_scores=replace_masked_values(bd_scores,mask_fi,-1e20)
        dummy_dims=[bd_scores.size(0),bd_scores.size(1),1]
        dummy_scores=bd_scores.new_zeros(*dummy_dims)
        bd_scores=torch.cat((dummy_scores,bd_scores),-1)
        _,predicted_bd=bd_scores.max(2)
        output_dict={
            'tags':tags,
            'bd_scores':bd_scores,
            'predicted_bd':predicted_bd
        }

        if tags is not None:
            self._bd_metrics(predicted_bd,tags,mask)
            bd_scores_flat=bd_scores.view(-1,self._n_labels)
            bd_label_flat=tags.view(-1)
            mask_flat=torch.tensor(mask.view(-1),dtype=torch.uint8)
            loss=self._loss(bd_scores_flat[mask_flat],bd_label_flat[mask_flat])
            output_dict['loss']=loss
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        bd_precision, bd_recall, bd_f1 = self._bd_metrics.get_metric(reset)
        return {"bd_precision": bd_precision,
                "bd_recall": bd_recall,
                "bd_f1": bd_f1}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params, regularizer: RegularizerApplicator) -> "LayerNer":
        return cls(vocab=vocab, params=params, regularizer=regularizer)
