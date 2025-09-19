import pandas as pd
from ctgan import CTGAN
from sdmetrics.reports.single_table import QualityReport
from sdv.metadata import SingleTableMetadata
import json

# Load dataset
data = pd.read_csv("data/adult.csv", header=None)

# Add column names for Adult dataset
data.columns = [
    "Age", "Workclass", "Fnlwgt", "Education", "Education-Num",
    "Marital-Status", "Occupation", "Relationship", "Race", "Sex",
    "Capital-Gain", "Capital-Loss", "Hours-per-week", "Native-Country", "Income"
]

print("Data loaded. Shape:", data.shape)

# Train CTGAN model
discrete_columns = [
    "Workclass", "Education", "Marital-Status", "Occupation",
    "Relationship", "Race", "Sex", "Native-Country", "Income"
]

model = CTGAN(epochs=10)   # training with 10 epochs for testing
model.fit(data, discrete_columns=discrete_columns)

print("Model training complete!")

# Generate synthetic data
synthetic = model.sample(5000)
synthetic.to_csv("models/ctgan-preet/synthetic_adult.csv", index=False)
print("Synthetic dataset saved at models/ctgan-preet/synthetic_adult.csv")

# Detect metadata from real dataset
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)

# Generate quality report
report = QualityReport()
report.generate(
    real_data=data,
    synthetic_data=synthetic,
    metadata=metadata.to_dict()
)

# Get overall quality score
overall_score = report.get_score()
print(f"\nOverall Quality Score: {overall_score:.2f}")

# Get detailed report
details = {
    "overall_score": overall_score,
    "column_shapes": report.get_details(property_name="Column Shapes").to_dict(),
    "column_pair_trends": report.get_details(property_name="Column Pair Trends").to_dict(),
}

# Save quality report as JSON
with open("models/ctgan-preet/adult_quality_report.json", "w") as f:
    json.dump(details, f, indent=4)

print("Readable quality report saved at models/ctgan-preet/adult_quality_report.json")
