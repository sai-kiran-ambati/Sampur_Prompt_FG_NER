from datasets import load_dataset

# Option A: Download directly into a folder inside your VS Code project
ds = load_dataset("prachuryyaIITG/SampurNER", cache_dir="./sampurner_data")

# Option B: Save it in a clean, reloadable Arrow format inside your project
ds.save_to_disk("./sampurner_saved")

print(ds)
