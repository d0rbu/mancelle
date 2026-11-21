# Modified by nano-gpt-pretrain-steer: parse judge ratings strictly and keep
# undeclared steering cells unavailable instead of scoring them. See ../../VENDORED.md.

import asyncio
import logging
import math
import re

import numpy as np
import pandas as pd

from .evaluator import Evaluator
from .prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE,
)


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.WARN,
)
logger = logging.getLogger(__name__)

_RATING_FRAME = re.compile(
    r"(?<!\w)Rating:\s*\[\[\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*\]\]"
)


class JudgeRatingParseError(ValueError):
    """A judge completion does not contain one valid rating frame."""


def parse_judge_rating(completion, min_rating=0.0, max_rating=2.0):
    """Return the one finite ``Rating: [[number]]`` value in range."""

    if not isinstance(completion, str):
        raise JudgeRatingParseError(
            f"judge completion must be a string, got {type(completion).__name__}"
        )
    matches = list(_RATING_FRAME.finditer(completion))
    if len(matches) != 1 or completion.count("Rating:") != 1:
        raise JudgeRatingParseError(
            "judge completion must contain exactly one Rating: [[finite number]] frame"
        )
    rating = float(matches[0].group(1))
    if not math.isfinite(rating):
        raise JudgeRatingParseError(f"judge rating must be finite, got {rating!r}")
    if not min_rating <= rating <= max_rating:
        raise JudgeRatingParseError(
            f"judge rating {rating} is outside [{min_rating}, {max_rating}]"
        )
    return rating


def _generation_is_available(generation):
    if generation is None:
        return False
    missing = pd.isna(generation)
    if isinstance(missing, (bool, np.bool_)):
        return not bool(missing)
    raise ValueError("steered generation must be a scalar value")


def _expand(values, positions, size, unavailable):
    if len(values) != len(positions):
        raise RuntimeError(
            f"judge returned {len(values)} values for {len(positions)} declared cells"
        )
    expanded = [unavailable] * size
    for position, value in zip(positions, values):
        expanded[position] = value
    return expanded


class LMJudgeEvaluator(Evaluator):
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.lm_model = kwargs.get("lm_model", None)
        self.concept_id = kwargs.get("concept_id", None)
        self.steer_dataset_type = kwargs.get("steer_dataset_type", None)

    def __str__(self):
        return "LMJudgeEvaluator"

    def _get_rating_from_completion(self, completion):
        return parse_judge_rating(completion)

    def _get_ratings_from_completions(self, completions, min_rating=0.0, max_rating=2.0):
        return [
            parse_judge_rating(completion, min_rating, max_rating)
            for completion in completions
        ]

    def _get_ratings_from_prompts(self, prompts, api_name, min_rating=0.0, max_rating=2.0):
        if not prompts:
            return [], []

        def validate(completion):
            return parse_judge_rating(completion, min_rating, max_rating)

        async def process_batch():
            return await self.lm_model.chat_completions(
                f"{api_name}_{self.model_name}_LMJudgeEvaluator",
                prompts,
                batch_size=16,
                validator=validate,
            )

        completions = asyncio.run(process_batch())
        return self._get_ratings_from_completions(
            completions, min_rating, max_rating
        ), completions

    def _get_all_ratings_from_data(self, data, column_name):
        model_relevance_concept_prompts = []
        model_relevance_instruction_prompts = []
        model_fluency_prompts = []
        dataset_names = []
        declared_positions = []

        for position, (_, row) in enumerate(data.iterrows()):
            dataset_names.append(row["dataset_name"])
            generation = row[f"{column_name}_steered_generation"]
            if not _generation_is_available(generation):
                continue
            declared_positions.append(position)
            model_relevance_concept_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
                    concept=row["input_concept"], sentence=generation
                )
            )
            model_relevance_instruction_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_INSTRUCTION_RELEVANCE_TEMPLATE.format(
                    instruction=row["original_prompt"], sentence=generation
                )
            )
            model_fluency_prompts.append(
                UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
                    sentence=generation
                )
            )

        concept_ratings, concept_completions = self._get_ratings_from_prompts(
            model_relevance_concept_prompts, f"{column_name}_concept"
        )
        instruction_ratings, instruction_completions = self._get_ratings_from_prompts(
            model_relevance_instruction_prompts, f"{column_name}_instruction"
        )
        fluency_ratings, fluency_completions = self._get_ratings_from_prompts(
            model_fluency_prompts, f"{column_name}_fluency"
        )

        size = len(data)
        concept_prompts = _expand(
            model_relevance_concept_prompts, declared_positions, size, None
        )
        instruction_prompts = _expand(
            model_relevance_instruction_prompts, declared_positions, size, None
        )
        fluency_prompts = _expand(model_fluency_prompts, declared_positions, size, None)
        concept_ratings = _expand(concept_ratings, declared_positions, size, np.nan)
        instruction_ratings = _expand(instruction_ratings, declared_positions, size, np.nan)
        fluency_ratings = _expand(fluency_ratings, declared_positions, size, np.nan)
        concept_completions = _expand(
            concept_completions, declared_positions, size, None
        )
        instruction_completions = _expand(
            instruction_completions, declared_positions, size, None
        )
        fluency_completions = _expand(
            fluency_completions, declared_positions, size, None
        )

        return (
            list(zip(concept_prompts, concept_ratings)),
            list(zip(instruction_prompts, instruction_ratings)),
            list(zip(fluency_prompts, fluency_ratings)),
            concept_completions,
            instruction_completions,
            fluency_completions,
            dataset_names,
        )

    def compute_metrics(self, data, write_to_dir=None):
        """Compute concept, instruction, fluency, and harmonic-mean scores."""

        logger.warning(
            f"Starting task for concept_id: {self.concept_id}, "
            f"model: {self.model_name}, evaluator: {self.__str__()}"
        )
        data_copy = data.copy()

        (
            model_relevance_concept_ratings,
            model_relevance_instruction_ratings,
            model_fluency_ratings,
            model_relevance_concept_completions,
            model_relevance_instruction_completions,
            model_fluency_completions,
            dataset_names,
        ) = self._get_all_ratings_from_data(data_copy, self.model_name)

        all_relevance_concept_ratings = []
        all_relevance_instruction_ratings = []
        all_fluency_ratings = []
        all_aggregated_ratings = []

        for i in range(len(model_relevance_concept_ratings)):
            concept_rating = model_relevance_concept_ratings[i][-1]
            instruction_rating = model_relevance_instruction_ratings[i][-1]
            fluency_rating = model_fluency_ratings[i][-1]
            all_relevance_concept_ratings.append(concept_rating)
            all_relevance_instruction_ratings.append(instruction_rating)
            all_fluency_ratings.append(fluency_rating)

            if any(math.isnan(score) for score in (
                concept_rating, instruction_rating, fluency_rating
            )):
                all_aggregated_ratings.append(np.nan)
                continue
            if dataset_names[i] == "AlpacaEvalSuppress":
                model_scores = [
                    2 - concept_rating,
                    instruction_rating,
                    fluency_rating,
                ]
            else:
                model_scores = [concept_rating, instruction_rating, fluency_rating]
            if 0 in model_scores:
                model_score = 0
            else:
                model_score = len(model_scores) / sum(1 / score for score in model_scores)
            all_aggregated_ratings.append(model_score)

        def json_rating(value):
            return None if math.isnan(value) else value

        metrics = {
            "lm_judge_rating": [],
            "relevance_concept_ratings": [],
            "relevance_instruction_ratings": [],
            "fluency_ratings": [],
            "factor": [],
            "raw_relevance_concept_ratings": [
                json_rating(value) for value in all_relevance_concept_ratings
            ],
            "raw_relevance_instruction_ratings": [
                json_rating(value) for value in all_relevance_instruction_ratings
            ],
            "raw_fluency_ratings": [
                json_rating(value) for value in all_fluency_ratings
            ],
            "raw_aggregated_ratings": [
                json_rating(value) for value in all_aggregated_ratings
            ],
            "relevance_concept_completions": model_relevance_concept_completions,
            "relevance_instruction_completions": model_relevance_instruction_completions,
            "fluency_completions": model_fluency_completions,
        }
        data_copy[f"{self.model_name}_lm_judge_rating"] = all_aggregated_ratings
        data_copy[f"{self.model_name}_relevance_concept_ratings"] = (
            all_relevance_concept_ratings
        )
        data_copy[f"{self.model_name}_relevance_instruction_ratings"] = (
            all_relevance_instruction_ratings
        )
        data_copy[f"{self.model_name}_fluency_ratings"] = all_fluency_ratings

        judged_data = data_copy.loc[
            data_copy[f"{self.model_name}_lm_judge_rating"].notna()
        ]
        for factor, group in judged_data.groupby("factor"):
            metrics["lm_judge_rating"].append(
                group[f"{self.model_name}_lm_judge_rating"].mean()
            )
            metrics["relevance_concept_ratings"].append(
                group[f"{self.model_name}_relevance_concept_ratings"].mean()
            )
            metrics["relevance_instruction_ratings"].append(
                group[f"{self.model_name}_relevance_instruction_ratings"].mean()
            )
            metrics["fluency_ratings"].append(
                group[f"{self.model_name}_fluency_ratings"].mean()
            )
            metrics["factor"].append(factor)

        return metrics
