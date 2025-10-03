import kagglehub

# Download latest version
path = kagglehub.dataset_download("prajeshsanghvi/osic-pulmonary-fibrosis-progression")

print("Path to dataset files:", path)