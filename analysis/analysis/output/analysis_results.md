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
| Period         |   count |    mean |     std |   min |   25% |   50% |   75% |   max |
|:---------------|--------:|--------:|--------:|------:|------:|------:|------:|------:|
| 2020 and After |    2705 | 13.5331 | 200.352 |     0 |     0 |     0 |     1 |  9261 |
| Before 2020    |    2607 | 29.7461 | 210.659 |     0 |     0 |     1 |     2 |  5681 |

## Recent Spills Response Time Summary:
| Period         |   count |    mean |     std |   min |   25% |   50% |   75% |   max |
|:---------------|--------:|--------:|--------:|------:|------:|------:|------:|------:|
| 2020 and After |    4472 | 2.92688 | 27.3021 |     0 |     0 |     1 |     1 |  1329 |
| Before 2020    |    6206 | 4.71221 | 57.1161 |     0 |     1 |     1 |     2 |  2232 |

## T-Test Results for Historical Spills
T-Statistic: 2.8723
P-Value: 0.0041
The difference in response time for historical spills is statistically significant.

## T-Test Results for Recent Spills
T-Statistic: 2.1457
P-Value: 0.0319
The difference in response time for recent spills is statistically significant.
