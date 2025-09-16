# ---
# jupytext:
#   formats: ipynb,py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
#     format_version: '1.3'
#     jupytext_version: 1.16.0
# kernelspec:
#   display_name: DL 2025
#   language: python
#   name: python3
# ---

# %% [markdown]
# # Week XX · Replace with Title
# Use this notebook as the starting point for the week's live session and exercises.

# %% [markdown]
# ## Objectives
# - [ ] Describe the primary learning goals for this week.
# - [ ] List any datasets, pretrained models, or external readings.
# - [ ] Update before publishing to students.

# %%
"""Environment verification."""
from __future__ import annotations

import keras
import tensorflow as tf

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
assert tuple(int(x) for x in tf.__version__.split(".")[:2]) >= (2, 0), "TensorFlow 2+ is required."

# %% [markdown]
# ## Warm-up
# - Add a short recap from last week or an opening question.
# - Provide a starter code cell or link to prerequisite reading if helpful.

# %%
"""Main lesson code goes here."""
# TODO: implement the walkthrough, demos, or experiments for this week.

# %% [markdown]
# ## Homework / Exercises
# - Draft the homework tasks, datasets, and submission expectations here.
# - Remember to push solutions to the `solutions/` branch only if required.
