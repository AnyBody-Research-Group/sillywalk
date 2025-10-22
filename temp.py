import polars as pl

from sillywalk import PCAPredictor
from sillywalk.anybody import write_anyscript

data = pl.read_csv("tests/Fourier.csv")

model_gait = PCAPredictor(data, n_components=0.99)

constraints = {"Speed": 1.2, "Height": 1.4}
result = model_gait.predict(constraints)

print(list(result[k] for k in constraints))

write_anyscript(result, "walking.main.any", create_human_model=True)

# The model can be saved to a .npz file, which only stores the means, std, and eigenvalues/vectors
# This reduces the size compared to the full csv dataset. In this case 31 MB -> 1 MB
model_gait.export_pca_data("walking.npz")

# The model can then be loadeded directly from the npz file:
model_gait_loaded = PCAPredictor.from_pca_data("walking.npz")
