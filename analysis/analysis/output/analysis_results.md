# Analysis Results

## Historical Spills Report Delay Summary:
|       |   Report Delay (Days) |
|:------|----------------------:|
| count |              5312     |
| mean  |                21.49  |
| std   |               205.616 |
| min   |                 0     |
| 25%   |                 0     |
| 50%   |                 1     |
| 75%   |                 2     |
| max   |              9261     |

## Recent Spills Report Delay Summary:
|       |   Report Delay (Days) |
|:------|----------------------:|
| count |           10678       |
| mean  |               3.96451 |
| std   |              46.9977  |
| min   |               0       |
| 25%   |               0       |
| 50%   |               1       |
| 75%   |               2       |
| max   |            2232       |

## Historical Spills Response Time Summary:
| Period         |   count |    mean |      std |   min |   25% |   50% |   75% |   max |
|:---------------|--------:|--------:|---------:|------:|------:|------:|------:|------:|
| 2021 and After |    2451 | 10.1816 |  96.0115 |     0 |     0 |     0 |     1 |  2501 |
| Before 2021    |    2861 | 31.1779 | 265.348  |     0 |     0 |     1 |     2 |  9261 |

## Recent Spills Response Time Summary:
| Period         |   count |    mean |     std |   min |   25% |   50% |   75% |   max |
|:---------------|--------:|--------:|--------:|------:|------:|------:|------:|------:|
| 2021 and After |    3552 | 2.92877 | 28.8645 |     0 |     0 |     1 |     1 |  1329 |
| Before 2021    |    7126 | 4.48077 | 53.7949 |     0 |     0 |     1 |     2 |  2232 |

## T-Test Results for Historical Spills
T-Statistic: 3.9419
P-Value: 0.0001
The difference in response time for historical spills is statistically significant.

## T-Test Results for Recent Spills
T-Statistic: 1.9390
P-Value: 0.0525
The difference in response time for recent spills is not statistically significant.
