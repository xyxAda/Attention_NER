# coding: utf-8

import logging
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util

from smhsan.training.metrics import TypeF1Measure

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


# Mapping specific to the dataset used in our setting (ACE2005)
# Please adapt it if necessary
#rel_type_2_idx = {"O":0,"GPE":1, "ORG":2, "PER":3, "FAC":4, "VEH":5, "LOC":6, "WEA":7}
rel_type_2_idx={"O":0,"DNA":1, "RNA":2, "protein":3, "cell_line":4, "cell_type":5}
idx_2_rel_type = {value: key for key, value in rel_type_2_idx.items()}


@Model.register("type_extractor")
class TypeExtractor(Model):
    """
	A class containing the scoring model for Type extraction.

	Parameters
	----------
	vocab: ``allennlp.data.Vocabulary``, required.
        The vocabulary fitted on the data.
	text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``text`` ``TextField`` we get as input to the model.
    context_layer : ``Seq2SeqEncoder``, required
        This layer incorporates contextual information for each word in the document.
	d: ``int``, required
		The (half) dimension of embedding given	by the encoder context_layer.
	l: ``int``, required
		The dimension of the types extractor scorer embedding.
	n_classes: ``int``, required
		The number of different possible type classes.
	activation: ``str``, optional (default = "relu")
		Non-linear activation function for the scorer. Can be either "relu" or "tanh".
	"""

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        context_layer: Seq2SeqEncoder,
        d: int,
        l: int,
        n_classes: int,
        threshold:float,
        dropout: Optional[float] = None,
        activation: str = "relu",
        label_namespace: str = "json_ec_labels",
    ) -> None:
        super(TypeExtractor, self).__init__(vocab)

        self._Wh = nn.Parameter(torch.Tensor(2 * d, l))
        self._Wt = nn.Parameter(torch.Tensor(2 * d, l))
        self._Wr = nn.Parameter(torch.Tensor(l, n_classes))
        #self._bh = nn.Parameter(torch.Tensor(l))
        #self._bt = nn.Parameter(torch.Tensor(l))
        self._b = nn.Parameter(torch.Tensor(l))

        self.init_weights()

        self._n_classes = n_classes
        self._activation = activation
        self._threshold=threshold

        self._text_field_embedder = text_field_embedder
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        self._context_layer = context_layer

        self._label_namespace = label_namespace

        self._type_metric = TypeF1Measure()

        self._loss_fn_0 = nn.BCEWithLogitsLoss()
        self._loss_fn_1=nn.BCELoss()

    def init_weights(self) -> None:
        """
		Initialization for the weights of the model.
		"""
        nn.init.kaiming_normal_(self._Wh)
        nn.init.kaiming_normal_(self._Wt)
        nn.init.kaiming_normal_(self._Wr)

        #nn.init.normal_(self._bh)
        #nn.init.normal(self._bt)
        nn.init.normal(self._b)

    def multi_class_cross_entropy_loss(self, scores, labels, mask):
        """
		Compute the loss from
		"""
        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        padded_document_length = mask.size(1)
        mask = mask.float()  # Size: n_batches x padded_document_length
        squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0)
        squared_mask = squared_mask.unsqueeze(-1).repeat(
            1, 1, 1, self._n_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        # The scores (and gold labels) are flattened before using
        # the binary cross entropy loss.
        flat_size = scores.size()
        scores = scores * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        #

        scores_flat = scores.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)
        labels = labels * squared_mask  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        labels_flat = labels.view(
            flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)
        #pdb.set_trace()
        loss = self._loss_fn_0(scores_flat, labels_flat)

        # Amplify the loss to actually see something...
        del squared_mask
        return 100 * loss

    @overrides
    def forward(self, text: Dict[str, torch.LongTensor], types: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
		Forward pass of the model.
		Compute the predictions and the loss (if labels are available).

		Parameters:
		----------
		text: Dict[str, torch.LongTensor]
			The input sentences which have transformed into indexes (integers) according to a mapping token:str -> token:int
		types: torch.IntTensor
			The gold types to predict.
		"""

        text_embeddings = self._text_field_embedder(text)
        mask = util.get_text_field_mask(text)
        if self.dropout:
            text_embeddings = self.dropout(text_embeddings)


        # Compute the contextualized representation from the word embeddings.
        # Usually, _context_layer is a Seq2seq model such as LSTM
        encoded_text = self._context_layer(
            text_embeddings, mask
        )  # Size: batch_size x padded_document_length x lstm_output_size

        ###### Type Scorer ##############
        # Compute the type scores
        left = torch.matmul(encoded_text, self._Wh)  # Size: batch_size x padded_document_length x l
        #left=left+self._bh
        right = torch.matmul(encoded_text, self._Wt)  # Size: batch_size x padded_document_length x l
        #right=right+self._bt

        left = left.permute(1, 0, 2)
        left = left.unsqueeze(3)
        right = right.permute(0, 2, 1)
        right = right.unsqueeze(0)

        B = left + right
        B = B.permute(1, 0, 3, 2)  # Size: batch_size x padded_document_length x padded_document_length x l

        #outer_sum_bias=B
        outer_sum_bias = B+self._b
        if self._activation == "relu":
            activated_outer_sum_bias = F.relu(outer_sum_bias)
        elif self._activation == "tanh":
            activated_outer_sum_bias = F.tanh(outer_sum_bias)

        type_scores = torch.matmul(
            activated_outer_sum_bias, self._Wr
        )  # Size: batch_size x padded_document_length x padded_document_length x n_classes
        #################################################################

        batch_size, padded_document_length = mask.size()
        type_sigmoid_scores = torch.sigmoid(
            type_scores
        )   #Size: batch_size x padded_document_length x padded_document_length x n_classes

        # predicted_types[l, i, j, k] == 1 iif we predict a type k with ARG1==i, ARG2==j in the l-th sentence of the batch
        typeCheck=torch.zeros_like(type_sigmoid_scores)+self._threshold
        predicted_types=torch.gt(type_sigmoid_scores,typeCheck).float()



        output_dict = {
            "type_sigmoid_scores": type_sigmoid_scores,
            "predicted_types": predicted_types,
            "mask": mask,
        }
        #pdb.set_trace()
        if types is not None:
            # Size: batch_size x padded_document_length x padded_document_length x n_classes
            # gold_types[l, i, j, k] == 1 iif we predict a type k with ARG1==i, ARG2==j in the l-th sentence of the batch
            gold_types = torch.zeros(batch_size, padded_document_length, padded_document_length, self._n_classes)

            for exple_idx, exple_tags in enumerate(types):  # going through the batch
                # encodes a type in the sentence where the two non zeros elements
                # indicate the two words arguments AND the type between these two words.
                for rel in exple_tags:
                    if rel.sum().item() == 0:
                        continue

                    for idx in rel.nonzero():
                        label_srt = self.vocab.get_token_from_index(rel[idx].item(), self._label_namespace)
                        if '__' in label_srt:
                            arg, rel_type = label_srt.split("__")
                            if arg == "ARG1":
                                x = idx.data[0]
                            else:
                                y = idx.data[0]
                        else:
                            rel_type=label_srt
                            x=y=idx.data[0]

                    gold_types[exple_idx, x, y, rel_type_2_idx[rel_type]] = 1

                    # GPU support
            if text_embeddings.is_cuda:
                gold_types = gold_types.cuda(device=type_scores.device)

            # Compute the loss
            output_dict["loss"] = self.multi_class_cross_entropy_loss(
                scores=type_scores, labels=gold_types, mask=mask
            )

            # Compute the metrics with the predictions.
            self._type_metric(predictions=predicted_types, gold_labels=gold_types, mask=mask)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
		Decode the predictions
		"""
        decoded_predictions = []

        for instance_tags in output_dict["predicted_types"]:
            sentence_length = instance_tags.size(0)
            decoded_types = []

            for arg1, arg2, rel_type_idx in instance_tags.nonzero().data:
                type = ["*"] * sentence_length
                rel_type = idx_2_rel_type[rel_type_idx.item()]
                type[arg1] = "ARG1__" + rel_type
                type[arg2] = "ARG2__" + rel_type
                decoded_types.append(type)

            decoded_predictions.append(decoded_types)

        output_dict["decoded_predictions"] = decoded_predictions

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
		Compute the metrics for type: precision, recall and f1.
		A type is considered correct if we can correctly predict the word of ARG1, the word of ARG2 and the type.
		"""
        metric_dict = self._type_metric.get_metric(reset=reset)
        return {x: y for x, y in metric_dict.items() if "overall" in x}
