"""
remove spaces in csv file.

run: python clean_data.py
"""

print('Processing')

with open("census.csv", 'r') as fr:
    with open('clean_census.csv', 'w') as fw:
        lines = fr.readlines()
        lines = [line.replace(" ", "") for line in lines]
        fw.writelines(lines)

print('Processed!')
