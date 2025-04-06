import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

def main():
    data = pd.read_csv("data.csv")

    mean_times = data.groupby("sorting_function")["time_taken"].mean().sort_values(ascending=True)
    print(mean_times)

    sorting_functions = mean_times.index.tolist()

    p_values = [
        (sort1, sort2, ttest_ind(
            data[data["sorting_function"] == sort1]["time_taken"],
            data[data["sorting_function"] == sort2]["time_taken"],
            equal_var=False
        )[1])
        for i, sort1 in enumerate(sorting_functions)
        for sort2 in sorting_functions[i + 1:]
    ]

    adjusted_pvals = multipletests([p[2] for p in p_values], method='bonferroni')[1]
    indistinguishable = [(s1, s2) for (s1, s2, _), p in zip(p_values, adjusted_pvals) if p >= 0.05]


    print(pairwise_tukeyhsd(data["time_taken"], data["sorting_function"], alpha=0.05))
    if indistinguishable:
        print("\nIndistinguishable:", ", ".join(f"{s1} & {s2}" for s1, s2 in indistinguishable))

if __name__ == "__main__":
    main()
