# Tharun Gopi
import csv

# open the file in read mode
filename = open('portland_housing.csv', 'r')

# creating dictated object
file = csv.DictReader(filename)

# creating empty lists
sizes = []
prices = []

# iterating over each row and append
# values to empty list
for col in file:
    sizes.append(col['size'])
    prices.append(col['price'])

# printing lists
print('Sizes:', sizes)
print('Prices:', prices)
