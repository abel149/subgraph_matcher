import pickle

with open("directed.pkl", "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data.keys() if isinstance(data, dict) else "Not a dict")
