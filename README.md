# Hacker Statistics in Python

A Jupyter notebook to hack on statistics with Python, adapted from DataCamp and Wikipedia.

## Definitions

### Variance

The standard deviation is the mean squared distance of the data from their mean. Informally, a measure of the spread of data.

```python
import numpy as np
var = np.var(df['column1'])
```

### Standard Deviation

The standard deviation is a measure that is used to quantify the amount of variation or dispersion of a set of data values.

```python
import numpy as np
var1 = np.std(df['column1'])
var2 = np.sqrt(np.var(df['column1']))
```

### Covariance

The covariance is a measure of the joint variability of two random variables. Informally, a measure of how two quantities vary together.

```python
import numpy as np
cov = np.cov(df['column1'], df['column2'])
```

## Functions

### Probability Mass Function (PMF)

A probability mass function (PMF) is a function that gives the probability that a discrete random variable is exactly equal to some value.

### Probability Density Function (PDF)

A probability density function (PDF) is a function whose value at any given sample in the sample space can be interpreted as providing a relative likelihood that the value of the random variable would equal that sample.

### Cumulative Distribution Function (CDF)

A cumulative distribution function (CDF) of a real-valued random variable X, or just distribution function of X, evaluated at x, is the probability that X will take a value less than or equal to x.

### Empirical Cumulative Distribution Function (ECDF)

An Empirical Cumulative Distribution Function (ECDF) is a step function that jumps up by 1/n at each of the n data points.

```python
import numpy as np
x = np.sort(df['column1'])
y = np.arange(1, len(x)+1) / len(x)
```

### Pearson Correlation

The Pearson correlation coefficient is a measure of the linear correlation between two variables X and Y. It has a value between +1 and −1, where 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total negative linear correlation.

## Distributions

### Binomial Distribution

The binomial distribution with parameters n and p is the discrete probability distribution of the number of successes in a sequence of n independent experiments, each asking a yes–no question, and each with its own boolean-valued outcome: a random variable containing a single bit of information: success/yes/true/one (with probability p) or failure/no/false/zero (with probability q = 1 − p).

### Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate and independently of the time since the last event.

### Exponential Distribution

The exponential distribution is the continuous probability distribution that describes the time between events in a Poisson point process.

### Normal Distribution

The normal distribution states, under some conditions (which include finite variance), that averages of samples of observations of random variables independently drawn from independent distributions converge in distribution to the normal, that is, become normally distributed when the number of observations is sufficiently large.
