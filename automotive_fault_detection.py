"""
Automotive Sensor Fault Detection with Machine Learning

This project creates simulated automotive sensor data and trains a machine
learning model to detect different vehicle fault conditions.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
DATA_FILE = DATA_DIR / "vehicle_sensor_data.csv"
REPORT_FILE = OUTPUT_DIR / "ml_report.txt"


def create_sample_data() -> pd.DataFrame:
    """Create simulated automotive sensor data."""

    data = [
        # speed, rpm, battery, temp, brake, throttle, vibration, fault
        [50, 1800, 12.6, 70, 5, 30, 0.2, "normal"],
        [60, 2100, 12.5, 75, 4, 35, 0.3, "normal"],
        [80, 2600, 12.4, 82, 6, 45, 0.4, "normal"],
        [100, 3200, 12.3, 88, 7, 55, 0.5, "normal"],
        [40, 1600, 12.7, 68, 3, 25, 0.2, "normal"],
        [70, 2500, 12.2, 98, 5, 40, 0.6, "high_temperature"],
        [90, 3000, 12.1, 105, 6, 50, 0.8, "high_temperature"],
        [110, 3600, 12.0, 112, 8, 60, 1.0, "high_temperature"],
        [65, 2300, 12.2, 101, 5, 42, 0.7, "high_temperature"],
        [45, 1700, 11.7, 72, 4, 28, 0.3, "low_battery"],
        [55, 2000, 11.4, 74, 5, 34, 0.4, "low_battery"],
        [75, 2500, 11.2, 80, 6, 45, 0.5, "low_battery"],
        [35, 1500, 11.6, 69, 3, 22, 0.3, "low_battery"],
        [30, 5000, 12.5, 77, 30, 10, 2.5, "sensor_anomaly"],
        [120, 900, 12.4, 79, 2, 95, 2.0, "sensor_anomaly"],
        [10, 6000, 12.6, 85, 40, 5, 3.0, "sensor_anomaly"],
        [140, 1000, 12.3, 90, 1, 100, 2.8, "sensor_anomaly"],
    ]

    columns = [
        "vehicle_speed_kmh",
        "engine_rpm",
        "battery_voltage_v",
        "motor_temperature_c",
        "brake_pressure_bar",
        "throttle_percent",
        "vibration_level",
        "fault_type",
    ]

    return pd.DataFrame(data, columns=columns)


def save_sample_data(data: pd.DataFrame) -> None:
    """Save the simulated data to a CSV file."""

    DATA_DIR.mkdir(exist_ok=True)
    data.to_csv(DATA_FILE, index=False)


def train_model(data: pd.DataFrame):
    """Train a Random Forest machine learning model."""

    x = data.drop(columns=["fault_type"])
    y = data["fault_type"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return model, x_test, y_test, predictions


def create_confusion_matrix(y_test, predictions) -> None:
    """Create and save a confusion matrix plot."""

    OUTPUT_DIR.mkdir(exist_ok=True)

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation=45,
    )

    plt.title("Fault Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
    plt.close()


def create_feature_importance_plot(model, feature_names) -> None:
    """Create and save a feature importance plot."""

    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png")
    plt.close()


def generate_report(y_test, predictions) -> None:
    """Generate a text report with model evaluation results."""

    report = classification_report(y_test, predictions)

    report_text = f"""
Automotive Sensor Fault Detection Report

This report shows the performance of a machine learning model trained to
classify vehicle sensor conditions.

Fault classes:
- normal
- high_temperature
- low_battery
- sensor_anomaly

Model:
Random Forest Classifier

Classification Report:

{report}
"""

    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_FILE.write_text(report_text.strip(), encoding="utf-8")


def main() -> None:
    """Run the complete machine learning workflow."""

    data = create_sample_data()
    save_sample_data(data)

    model, x_test, y_test, predictions = train_model(data)

    create_confusion_matrix(y_test, predictions)
    create_feature_importance_plot(model, x_test.columns)
    generate_report(y_test, predictions)

    print("Machine learning analysis completed successfully.")
    print(f"Dataset saved in: {DATA_FILE}")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()