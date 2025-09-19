import pandas as pd
from ctgan import CTGAN
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import json

# Load dataset
data = pd.read_csv("data/adult.csv", header=None)

# Add column names
data.columns = [
    "Age", "Workclass", "Fnlwgt", "Education", "Education-Num",
    "Marital-Status", "Occupation", "Relationship", "Race", "Sex",
    "Capital-Gain", "Capital-Loss", "Hours-per-week", "Native-Country", "Income"
]

print("Data loaded. Shape:", data.shape)

# Define categorical columns
discrete_columns = [
    "Workclass", "Education", "Marital-Status", "Occupation",
    "Relationship", "Race", "Sex", "Native-Country", "Income"
]

# Train CTGAN (longer training)
model = CTGAN(epochs=50)  
model.fit(data, discrete_columns=discrete_columns)

print("Model training complete with 50 epochs!")

# Generate synthetic samples
synthetic = model.sample(5000)
synthetic.to_csv("models/ctgan-preet/synthetic_adult_long.csv", index=False)
print("Synthetic dataset saved at models/ctgan-preet/synthetic_adult_long.csv")

# Metadata + Evaluation
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

report = QualityReport()
report.generate(
    real_data=data,
    synthetic_data=synthetic,
    metadata=metadata.to_dict()
)

# Print overall score
overall_score = report.get_score()
print(f"\nOverall Quality Score (50 epochs): {overall_score:.2f}")

# Save results to JSON
details = {
    "overall_score": overall_score,
    "column_shapes": report.get_details(property_name="Column Shapes").to_dict(),
    "column_pair_trends": report.get_details(property_name="Column Pair Trends").to_dict(),
}

with open("models/ctgan-preet/adult_quality_report_long.json", "w") as f:
    json.dump(details, f, indent=4)

print("Long training quality report saved at models/ctgan-preet/adult_quality_report_long.json")
