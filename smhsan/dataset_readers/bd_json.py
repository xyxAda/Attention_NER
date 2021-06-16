# coding: utf-8

import logging
from typing import Dict, List, Iterable
from overrides import overrides
import joblib

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

from allennlp.data.fields import Field, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bd_json")
class BDJsonReader(DatasetReader):

    def __init__(
            self,
            token_indexers: Dict[str, TokenIndexer] = None,
            domain_identifier: str = None,
            label_namespace: str = None,
            lazy: bool = False,
            constrain_crf_decoding: bool=True

    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._domain_identifier = domain_identifier
        self._label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache
        logger.info("Reading Boundary Detection instances from dataset files at: %s", file_path)
        if self._domain_identifier is not None:
            logger.info("Filtering to only include file paths containing the %s domain", self._domain_identifier)

        sentences,records=joblib.load(file_path)
        for sentence,record in zip(sentences,records):
            tagger=self.conver_bd_to_BIEO(len(sentence),record)
            tokens = [Token(t) for t in sentence]
            yield self.text_to_instance(tokens, tagger)

    def text_to_instance(self, tokens: List[Token], tags: List[str] = None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields["tokens"] = text_field
        if tags:
            fields["tags"] = SequenceLabelField(
                labels=tags, sequence_field=text_field, label_namespace=self._label_namespace
            )
        return Instance(fields)

    def conver_bd_to_BIEO(self,length,records):
        labels = ['O'] * length
        for record in records:
            for i in range(record[0] + 1, record[1] - 1):
                if labels[i] == 'B' or labels[i] == 'E':
                    continue
                labels[i] = 'I'
            labels[record[1] - 1] = 'E'
            labels[record[0]] = 'B'
        return labels







