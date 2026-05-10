import sys
import pandas as pd

output_filename = sys.argv[1]
results_files = sys.argv[2:]



results = [pd.read_csv(file) for file in results_files]

if "labels_flat_preprocessed" in results_files[2]:
    methods_names = [file[file.index("preprocessed_") + len("preprocessed_"): file.index("_inductive")] for file in results_files]
else:
    methods_names = [file[file.index("labels_final_") + len("labels_final_"): file.index("_inductive")] for file in results_files]


mean_results = pd.DataFrame([result.iloc[-1] for result in results])
mean_results.insert(0, "method", methods_names)
mean_results = mean_results.drop("folds",axis=1).sort_values(by = "method").round(3)

mean_results.to_csv(output_filename, index = False)
