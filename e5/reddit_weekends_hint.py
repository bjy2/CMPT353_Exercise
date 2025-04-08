import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, normaltest, levene, mannwhitneyu
import matplotlib.pyplot as plt

OUTPUT_TEMPLATE = (
    "Initial T-test p-value: {initial_ttest_p:.3g}\n"
    "Original data normality p-values: {initial_weekday_normality_p:.3g} {initial_weekend_normality_p:.3g}\n"
    "Original data equal-variance p-value: {initial_levene_p:.3g}\n"
    "Transformed data normality p-values: {transformed_weekday_normality_p:.3g} {transformed_weekend_normality_p:.3g}\n"
    "Transformed data equal-variance p-value: {transformed_levene_p:.3g}\n"
    "Weekly data normality p-values: {weekly_weekday_normality_p:.3g} {weekly_weekend_normality_p:.3g}\n"
    "Weekly data equal-variance p-value: {weekly_levene_p:.3g}\n"
    "Weekly T-test p-value: {weekly_ttest_p:.3g}\n"
    "Mann-Whitney U-test p-value: {utest_p:.3g}"
)


def main():
    # Read the input file
    reddit_counts = sys.argv[1]
    counts = pd.read_json(reddit_counts, lines=True)
    counts['date'] = pd.to_datetime(counts['date'])

    # Filter for 2012 and 2013 and /r/canada subreddit
    counts = counts[(counts['date'].dt.year.isin([2012, 2013])) & (counts['subreddit'] == 'canada')]

    # Classify days as weekends or weekdays
    counts['week_type'] = counts['date'].dt.weekday.apply(lambda x: 'weekend' if x >= 5 else 'weekday')

    # Separate weekday and weekend data
    weekdays = counts[counts['week_type'] == 'weekday']['comment_count']
    weekends = counts[counts['week_type'] == 'weekend']['comment_count']

    # Initial T-test
    initial_ttest_p = ttest_ind(weekdays, weekends, equal_var=False).pvalue

    # Normality test
    initial_weekday_normality_p = normaltest(weekdays).pvalue
    initial_weekend_normality_p = normaltest(weekends).pvalue

    # Equal variance test
    initial_levene_p = levene(weekdays, weekends).pvalue

    # Data transformation using sqrt
    transformed_weekdays = np.sqrt(weekdays)
    transformed_weekends = np.sqrt(weekends)
    transformed_weekday_normality_p = normaltest(transformed_weekdays).pvalue
    transformed_weekend_normality_p = normaltest(transformed_weekends).pvalue
    transformed_levene_p = levene(transformed_weekdays, transformed_weekends).pvalue

    # Fix 2: Central Limit Theorem - Aggregate by week and year
    counts['year_week'] = counts['date'].apply(lambda x: (x.isocalendar()[0], x.isocalendar()[1]))
    weekly_means = counts.groupby(['year_week', 'week_type'])['comment_count'].mean().reset_index(name='week_mean')
    weekday_means = weekly_means[weekly_means['week_type'] == 'weekday']['week_mean']
    weekend_means = weekly_means[weekly_means['week_type'] == 'weekend']['week_mean']
    weekly_weekday_normality_p = normaltest(weekday_means).pvalue
    weekly_weekend_normality_p = normaltest(weekend_means).pvalue
    weekly_levene_p = levene(weekday_means, weekend_means).pvalue
    weekly_ttest_p = ttest_ind(weekday_means, weekend_means).pvalue

    # Fix 3: Non-parametric test
    utest_p = mannwhitneyu(weekdays, weekends, alternative='two-sided').pvalue

    # Print results
    print(OUTPUT_TEMPLATE.format(
        initial_ttest_p=initial_ttest_p,
        initial_weekday_normality_p=initial_weekday_normality_p,
        initial_weekend_normality_p=initial_weekend_normality_p,
        initial_levene_p=initial_levene_p,
        transformed_weekday_normality_p=transformed_weekday_normality_p,
        transformed_weekend_normality_p=transformed_weekend_normality_p,
        transformed_levene_p=transformed_levene_p,
        weekly_weekday_normality_p=weekly_weekday_normality_p,
        weekly_weekend_normality_p=weekly_weekend_normality_p,
        weekly_levene_p=weekly_levene_p,
        weekly_ttest_p=weekly_ttest_p,
        utest_p=utest_p,
    ))


if __name__ == '__main__':
    main()
