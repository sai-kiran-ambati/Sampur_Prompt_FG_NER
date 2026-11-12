from datasets import load_dataset

telugu_ds = load_dataset(
    "prachuryyaIITG/SampurNER",
    data_files={
        "train": "SampurNER_te_train.txt",
        "validation": "SampurNER_te_dev.txt",
        "test": "SampurNER_te_test.txt",
    },
    cache_dir="./sampurner_telugu"
)

# Save the Telugu dataset to disk
telugu_ds.save_to_disk("./sampurner_telugu_saved")

print(telugu_ds)