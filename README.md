# ShakeIA

A shakespirian talking kind of IA.

## Data preparation and operations

### Data preparation

We started by preparing our data, by splitting the text on every characters, and remove extra whitespaces (e.g. cumulative spaces).

### Data vectors

We decided to use overlaping character vectors of a mean word size of 4 by default (exactly 4,79 using Norvig 2009 study on google words dataset), but this parameter can be changed as required.

The vector contains the index of the letter in a corresponding alphabet table composed of alpha_numeric characters, punction and whitespaces, totaling to 100 chars from 0 to 99.

### Dataset splitting

As we are working on sentences on a character split level, we cannot shuffle our dataset, as it would loose it semantic meaning.
For this we decided to only split on indexes with the proportion 7/1,5/1,5.

## Model

The model is based on a varying implementation of an *LTSM* adapted to work over chararcter vectors.

## Training operation

### Parameter optimisation

### Optuna and Adam

## Time and hardware

## Evaluation

### Dataset evaluation and validation

### Human evaluation
