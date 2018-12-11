# Hacker Statistics in Python

[![Binder](https://img.shields.io/badge/binder-iris.ipynb-green.svg)](https://mybinder.org/v2/gh/cbismuth/hacker-stats/master?filepath=iris.ipynb)

This repository contains:

* A [Jupyter notebook](iris.ipynb) to hack on statistics with Python (distributions, p-values ...)
* A [scikit-learn classifier](classifier.py) sample ([output](classifier.png))
* A [scikit-learn regression](regression.py) sample ([output](regression.png), [features](lasso.png))

# Definitions

Adapted from DataCamp and Wikipedia.

## Variance

The standard deviation is the mean squared distance of the data from their mean. Informally, a measure of the spread of data.

```python
import numpy as np
var = np.var(df['column1'])
```

## Standard Deviation

The standard deviation is a measure that is used to quantify the amount of variation or dispersion of a set of data values.

```python
import numpy as np
var1 = np.std(df['column1'])
var2 = np.sqrt(np.var(df['column1']))
```

## Covariance

The covariance is a measure of the joint variability of two random variables. Informally, a measure of how two quantities vary together.

```python
import numpy as np
cov = np.cov(df['column1'], df['column2'])
```

# Functions

## Probability Mass Function (PMF)

A probability mass function (PMF) is a function that gives the probability that a discrete random variable is exactly equal to some value.

## Probability Density Function (PDF)

A probability density function (PDF) is a function whose value at any given sample in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample.

## Cumulative Distribution Function (CDF)

A cumulative distribution function (CDF) of a real-valued random variable X, or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x.

## Empirical Cumulative Distribution Function (ECDF)

An Empirical Cumulative Distribution Function (ECDF) is a step function that jumps up by 1/n at each of the n data points.

```python
import numpy as np
x = np.sort(df['column1'])
y = np.arange(1, len(x)+1) / len(x)
```

## Pearson Correlation

The Pearson correlation coefficient is a measure of the linear correlation between two variables X and Y. It has a value between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.

## Confidence Interval

If we repeated measurements over and over again, p% of the observed values would lie within the p% confidence interval.

# Distributions

## Binomial Distribution

The binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments, each asking a yes–no question, and each with its own boolean-valued outcome: a random variable containing a single bit of information: success/yes/true/one (with probability p) or failure/no/false/zero (with probability q = 1 − p).

## Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event.

## Exponential Distribution

The exponential distribution is the continuous probability distribution that describes the time between events in a Poisson point process.

## Normal Distribution

The normal distribution states, under some conditions (which include finite variance), that averages of samples of observations of random variables independently drawn from independent distributions converge in distribution to the normal, that is, become normally distributed when the number of observations is sufficiently large.

# Boostrapping

## Simulation

Use mean of random choices to generate bootstrap replicates from a single distribution.

## One sample test

Use mean-shiftting when only one distribution and the mean of the other one are available.

```python
data_2 = data_1_shifted = data_1 - np.mean(data_1) + mean_2
```

Then use one of the "two sample test" methods below with `data_1` and `data_2`.

## Two sample test

Use permutation when the two distributions are available.

* Useful for correlation with Pearson correlation as `func`.
* Useful for A/B testing with difference fraction as `func`.

```python
# Null hypothesis: change has no effect between A and B.
# clickthrough_A, clickthrough_B: arr. of 1s and 0s
def generate_replicates(data_1: np.ndarray, data_2: np.ndarray, func: Callable[[np.ndarray, np.ndarray], float], size: int=1) -> np.ndarray:
    replicates = np.empty(size)
    for i in range(size):
        all = np.concatenate((data_1, data_2))
        permuted = np.random.permutation(all)
        sample_1 = permuted[:len(data_1)]
        sample_2 = permuted[len(data_2):]
        replicates[i] = func(sample_1, sample_2)
    return replicates


def diff_frac(data_A: np.ndarray, data_B: np.ndarray) -> float:
    frac_A = np.sum(data_A) / len(data_A)
    frac_B = np.sum(data_B) / len(data_B)
    return frac_B - frac_A


p = np.sum(diff_frac_replicates >= diff_frac_actual) / len(replicates)
```
