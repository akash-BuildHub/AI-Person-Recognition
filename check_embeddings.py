import pickle

# Open the embeddings file
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

# Check the type and content
print("Type of data:", type(data))
print("Data content:", data)
