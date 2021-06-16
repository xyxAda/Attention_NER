# coding: utf-8

import logging
from typing import Dict, List, Iterable, Iterator
from overrides import overrides
import joblib


from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, ListField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token






logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("ec_json")
class ECJsonReader(DatasetReader):

    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        label_namespace: str = None,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._label_namespace = label_namespace

    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)  # if `file_path` is a URL, redirect to the cache
        logger.info("Reading Entity types from dataset files at: %s", file_path)

        sentences,records=joblib.load(file_path)
        for sentence,record in zip(sentences,records):
            tokens=[Token(t) for t in sentence]

            types=self.conver_pair_to_type(len(tokens),record)
            if types==[]:
                types = None
                continue
            else:
                yield self.text_to_instance(tokens, types)


    def read_raw_data(self,datas):

        i = 0
        sentences = list()
        records = list()
        infer_record = dict()
        for data in datas:
            i += 1

            # print(data)
            words = data['context'].strip().split()

            # pdb.set_trace()
            record = data['span_position']
            entity_type = data['entity_label']
            for span in record:
                span_boundary = span.strip().split(',')
                infer_record[(int(span_boundary[0]), int(span_boundary[1]))] = entity_type
            if i % 7 == 0:
                sentences.append(words)
                records.append(infer_record)
                infer_record = dict()
        return sentences,records


    def text_to_instance(self, tokens: List[Token], types=None) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields["text"] = text_field
        if types is not None:
            field_list = []
            for type in types:
                field_list.append(
                    SequenceLabelField(
                        labels=type, sequence_field=text_field, label_namespace=self._label_namespace
                    )
                )
            fields["types"] = ListField(field_list=field_list)
        return Instance(fields)

    def conver_pair_to_type(self,length,record):
        types=[]
        for span,t in record.items():
            start=span[0]
            end=span[1]
            type = ['*' for _ in range(length)]
            if start==end-1:
                type[start]=t
            else:
                type[start] = 'ARG1__'+t
                type[end-1]='ARG2__'+t
            types.append(type)
        return types

