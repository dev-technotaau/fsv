"""Ninja Fence Staining — Genetic Algorithm model & hyperparameter search.

Two-stage GA:
  Stage 1 — model family search (categorical gene: which of 18 combos)
  Stage 2 — hyperparameter fine-tune GA on the winner from Stage 1

Entry point: python -m src.ga.cli
"""
__version__ = "0.1.0"
