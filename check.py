import pickle
import pprint

def inspect_pickle_file(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("Type of root object:", type(data))
    print("\nSample content preview:\n")
    
    # Pretty print the first few keys/items depending on the type
    if isinstance(data, dict):
        pprint.pprint({k: data[k] for k in list(data)[:5]})
    elif isinstance(data, list):
        pprint.pprint(data[:5])
    else:
        pprint.pprint(data)

if __name__ == "__main__":
    inspect_pickle_file("graph.pkl")
