"""The classification applications that the comparison run fits."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from models.ada_boost import MODEL_NOTE as ADA_NOTE
from models.ada_boost import build_estimator as build_ada
from models.decision_tree import MODEL_NOTE as TREE_NOTE
from models.decision_tree import build_estimator as build_tree
from models.extra_trees import MODEL_NOTE as EXTRA_NOTE
from models.extra_trees import build_estimator as build_extra
from models.gradient_boosting import MODEL_NOTE as BOOST_NOTE
from models.gradient_boosting import build_estimator as build_boost
from models.knn import MODEL_NOTE as KNN_NOTE
from models.knn import build_estimator as build_knn
from models.logistic_regression import MODEL_NOTE as LOGISTIC_NOTE
from models.logistic_regression import build_estimator as build_logistic
from models.naive_bayes import MODEL_NOTE as NB_NOTE
from models.naive_bayes import build_estimator as build_nb
from models.random_forest import MODEL_NOTE as FOREST_NOTE
from models.random_forest import build_estimator as build_forest
from models.svc import MODEL_NOTE as SVC_NOTE
from models.svc import build_estimator as build_svc


@dataclass(frozen=True)
class ModelSpec:
    name: str
    note: str
    build: Callable[[int], object]


def individual_models() -> list[ModelSpec]:
    return [
        ModelSpec("logistic_regression", LOGISTIC_NOTE, build_logistic),
        ModelSpec("decision_tree", TREE_NOTE, build_tree),
        ModelSpec("random_forest", FOREST_NOTE, build_forest),
        ModelSpec("extra_trees", EXTRA_NOTE, build_extra),
        ModelSpec("gradient_boosting", BOOST_NOTE, build_boost),
        ModelSpec("ada_boost", ADA_NOTE, build_ada),
        ModelSpec("svc", SVC_NOTE, build_svc),
        ModelSpec("knn", KNN_NOTE, build_knn),
        ModelSpec("naive_bayes", NB_NOTE, build_nb),
    ]


def comparison_models() -> list[ModelSpec]:
    from models.voting import MODEL_NOTE as VOTE_NOTE
    from models.voting import build_estimator as build_vote

    return [*individual_models(), ModelSpec("voting", VOTE_NOTE, build_vote)]
