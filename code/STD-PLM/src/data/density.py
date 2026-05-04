import csv
from collections import defaultdict

degree = defaultdict(int)
filepath = "../../../../data/traffic/PEMS08/PEMS08.csv"

with open(filepath, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    for row in reader:
        if len(row) < 2:
            continue

        i, j = int(row[0]), int(row[1])

        degree[i] += 1
        degree[j] += 1

num_nodes = len(degree)
num_edges = sum(degree.values()) // 2
avg_degree = sum(degree.values()) / num_nodes
density = num_edges / (num_nodes * (num_nodes - 1))

print("Nodes:", num_nodes)
print("Edges:", num_edges)
print("Average degree:", avg_degree)
print("Density:", density)