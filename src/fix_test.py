import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

full_data = pd.read_csv("data/full.csv", sep=':')
test_data = pd.read_csv("data/test.csv")

print(full_data.info())
print(test_data.info())

test_data.insert(1, "Survived", 0)


test_data['Survived'] = test_data.apply(
    lambda row: full_data.loc[full_data['name'] == row['Name']]['survived'].values[0],
    axis=1
)


test_data.to_csv("data/test_.csv", index=False)