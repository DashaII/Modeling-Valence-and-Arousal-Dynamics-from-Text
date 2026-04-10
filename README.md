# UKP_Psycontrol at SemEval-2026 Task 2

## Modeling Valence and Arousal Dynamics from Text

### Authors

**Darya Hryhoryeva¹²**, **Amaia Zurinaga³**, **Hamidreza Jamalabadi³**,
**Iryna Gurevych¹²**

¹ Ubiquitous Knowledge Processing Lab (UKP Lab), Technical University of
Darmstadt\
² National Research Center for Applied Cybersecurity ATHENE, Germany\
³ Psychiatric Control Systems Lab, Marburg University

------------------------------------------------------------------------

## Abstract

This paper presents our system developed for **SemEval-2026 Task 2**.
The task requires modeling both current affect and short-term affective
change in chronologically ordered user-generated texts.

We explore three complementary approaches:

1.  **LLM prompting** under user-aware and user-agnostic settings\
2.  A **pairwise Maximum Entropy (MaxEnt) model** with Ising-style
    interactions for structured transition modeling\
3.  A **lightweight neural regression model** incorporating recent
    affective trajectories and trainable user embeddings

Our findings indicate that LLMs effectively capture static affective
signals from text, whereas short-term affective variation in this
dataset is more strongly explained by recent numeric state trajectories
than by textual semantics.

Our system ranked **first among participating teams in both Subtask 1
and Subtask 2A** based on the official evaluation metric.

------------------------------------------------------------------------

# SemEval-2026 Task 2 Description

**Task Name:** Predicting Variation in Emotional Valence and Arousal
over Time from Ecological Essays

The task focuses on modeling subjectively experienced emotion from
longitudinal, self-reported data.

The dataset contains chronologically ordered essays and feeling-word
lists written by U.S. service-industry workers over several years. Each
entry is paired with self-assessed:

-   **Valence (0--4)**
-   **Arousal (0--2)**

The shared task includes:

-   **Subtask 1:** Longitudinal Affect Assessment
    -   Predict valence and arousal per text\
    -   Evaluated using Pearson correlation (r) and MAE\
    -   Between-user, within-user, and composite evaluation
-   **Subtask 2A:** Forecasting Future Variation
    -   Predict next-step changes in valence and arousal\
    -   Evaluated with user-level Pearson correlation and MAE

------------------------------------------------------------------------

# Data Overview

-   2,764 entries from 137 users\
-   Average 20 entries per user (median: 14)\
-   52% free-form essays\
-   48% feeling-word lists\
-   Data span seven two-week periods\
-   92% of users participated in only one period

Common data characteristics:

-   Invariant valence/arousal across some users\
-   Low-content or repetitive texts

------------------------------------------------------------------------

# System Overview

We address the subtasks using three approaches:

### 1. Maximum Entropy (MaxEnt) Model

A pairwise model with Ising-style interactions motivated by energy-based
mental state modeling.

### 2. LLM-Based Prompting

User-aware and user-agnostic prompting strategies to predict affect
directly from text.

### 3. Neural Regression Model

A lightweight model incorporating: - Recent affect trajectories -
Trainable user embeddings - Continuous emotion value prediction

------------------------------------------------------------------------

# Disclaimer
This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.


