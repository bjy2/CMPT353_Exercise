import sys

import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu

OUTPUT_TEMPLATE = (
    '"Did more/less users use the search feature?" p-value:  {more_users_p:.3g}\n'
    '"Did users search more/less?" p-value:  {more_searches_p:.3g} \n'
    '"Did more/less instructors use the search feature?" p-value:  {more_instr_p:.3g}\n'
    '"Did instructors search more/less?" p-value:  {more_instr_searches_p:.3g}'
)


def main():
    searchdata_file = sys.argv[1]

    data = pd.read_json(searchdata_file, orient='records', lines=True)

    data['odd'] = data['uid'] % 2 == 1
    data['even'] = data['uid'] % 2 == 0
    data['searched'] = data['search_count'] > 0
    data['instructor'] = data['is_instructor']

    contingency_all = pd.crosstab(data['even'], data['searched'])
    more_users_p= chi2_contingency(contingency_all)[1]

    more_searches_p = mannwhitneyu(
        data.loc[data['odd'], 'search_count'],
        data.loc[data['even'], 'search_count'],
    )[1]

    instructors = data[data['instructor']]

    contingency_instr = pd.crosstab(instructors['even'], instructors['searched'])
    more_instr_p = chi2_contingency(contingency_instr)[1]

    more_instr_searches_p = mannwhitneyu(
        instructors.loc[instructors['odd'], 'search_count'],
        instructors.loc[instructors['even'], 'search_count'],
    )[1]
    # ...

    # Output
    print(OUTPUT_TEMPLATE.format(
        more_users_p=more_users_p,
        more_searches_p=more_searches_p,
        more_instr_p=more_instr_p,
        more_instr_searches_p=more_instr_searches_p,
    ))


if __name__ == '__main__':
    main()
