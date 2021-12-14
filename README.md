# stats-thesis

Types of Confidence Intervals:

1. Invert two 1-sided tests
- invert one test using `alternative == "less"`, and another using `alternative == "greater"`
- for both tests, use `alpha/2`

2. Invert one 2-sided test
3. t-test confidence interval for difference in means
4. Bootstrap confidence interval

For types 1-3, we have two subtypes of confidence intervals for pooled vs. unpooled variances. Ultimately, we will have 7 types of confidence intervals to compare (since the bootstrap CI does not rely on assumptions of the population variances).

For each type of CI, simulate 7300 instances (generating new samples each time, but keeping the parameters the same). Then measure the coverage probability, i.e., the proportion of the 7300 CIs which capture the true difference in means (which is known, since the data is simulated).
