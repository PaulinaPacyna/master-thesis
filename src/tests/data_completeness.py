import os

from reading import Reading

reading = Reading()
categories = set(reading.categories.keys())
datasets = set(os.listdir("data/ts"))
concatenated = set(class_.split("_")[0] for class_ in reading.read_dataset()[1].ravel())


print("categories - datasets", categories - datasets)
print("datasets - categories", datasets - categories)
print("categories - concatenated", categories - concatenated)
print("concatenated - categories", concatenated - categories)
