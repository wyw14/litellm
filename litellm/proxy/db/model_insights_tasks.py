import json
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Final

from pydantic import ConfigDict, TypeAdapter

from litellm.constants import MODEL_INSIGHTS_DEFAULT_TASK, MODEL_INSIGHTS_TASK_TAG_PREFIX
from litellm.types.model_insights import ModelInsightTask

_TASKS_FILE: Final = Path(__file__).resolve().parent.parent / "model_insights_tasks.json"
_TASK_ENTRIES: Final = TypeAdapter(
    Mapping[str, Mapping[str, object]], config=ConfigDict(strict=True, hide_input_in_errors=True)
)


@lru_cache(maxsize=1)
def load_model_insight_tasks() -> dict[str, ModelInsightTask]:
    raw: Final = _TASK_ENTRIES.validate_python(json.loads(_TASKS_FILE.read_text()))
    return {name: ModelInsightTask.model_validate(dict(task_type=name, **entry)) for name, entry in raw.items()}


def model_insight_task_type(request_tags: Iterable[str]) -> str:
    return next(
        (
            task
            for tag in request_tags
            if tag.startswith(MODEL_INSIGHTS_TASK_TAG_PREFIX)
            and (task := tag.removeprefix(MODEL_INSIGHTS_TASK_TAG_PREFIX)) in load_model_insight_tasks()
        ),
        MODEL_INSIGHTS_DEFAULT_TASK,
    )
