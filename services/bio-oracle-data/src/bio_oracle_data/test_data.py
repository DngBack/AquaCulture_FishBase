import pyo_oracle

layers = pyo_oracle.list_layers()   # trả về dataframe-like list
# quick filter
[var for var in layers if "temp" in var.lower() or "thetao" in var.lower()]

# bounding box Vietnam (approx) - adjust if needed
constraints = {
    "latitude>=": 8.0,
    "latitude<=": 23.5,
    "longitude>=": 102.0,
    "longitude<=": 110.5
}
# example layer name pulled from list_layers
layer_name = "thetao_baseline_2000_2019_depthsurf"  # replace with actual
out = pyo_oracle.download_layers(layer_name, constraints=constraints)
print("Saved to:", out)