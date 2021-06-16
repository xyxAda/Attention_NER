# coding: utf-8


import tqdm
from typing import List, Dict, Any, Iterable
import torch

from allennlp.models.model import Model
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.common.checks import check_for_gpu
from allennlp.nn import util

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logger = logging.getLogger(__name__)


def evaluate(
    model: Model, instances: Iterable[Instance], task_name: str, data_iterator: DataIterator, cuda_device: int
) -> Dict[str, Any]:
    """
    Evaluate a model for a particular task (usually after training).

    """
    check_for_gpu(cuda_device)
    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances, num_epochs=1, shuffle=False)
        logger.info("Iterating over dataset")
        generator_tqdm = tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        eval_loss = 0
        nb_batches = 0
        for batch in generator_tqdm:
            batch = util.move_to_device(batch, cuda_device)
            nb_batches += 1

            eval_output_dict = model.forward(task_name=task_name, tensor_batch=batch)
            loss = eval_output_dict["loss"]
            eval_loss += loss.item()
            metrics = model.get_metrics(task_name=task_name)
            metrics["loss"] = float(eval_loss / nb_batches)

            description = ", ".join(["%s: %.2f" % (name, value) for name, value in metrics.items()]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        metrics = model.get_metrics(task_name=task_name, reset=True, full=True)
        metrics["loss"] = float(eval_loss / nb_batches)
        return metrics

