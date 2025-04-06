import time
import numpy as np
import pandas as pd
from implementations import all_implementations

def main():
    num_runs = 100
    array_size = 10000

    results = []

    for i in range(num_runs):
        random_array = np.random.randint(1, 100000, size=array_size)
        for sort in all_implementations:
            st = time.time()
            res = sort(random_array)
            en = time.time()

            results.append({
                "run": i + 1,
                "sorting_function": sort.__name__,
                "time_taken": en - st
            })

    df = pd.DataFrame(results)
    df.to_csv("data.csv", index=False)

if __name__ == "__main__":
    main()
