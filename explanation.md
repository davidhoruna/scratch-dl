import kagglehub

# Download latest version
path = kagglehub.dataset_download("data/fruits")

print("Path to dataset files:", path)