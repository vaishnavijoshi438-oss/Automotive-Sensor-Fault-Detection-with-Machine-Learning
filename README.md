# Automotive Sensor Fault Detection with Machine Learning

This project uses Python and machine learning to detect fault conditions in simulated automotive sensor data.

The goal is to demonstrate how vehicle sensor measurements can be analyzed to classify normal behavior and possible technical faults such as high temperature, low battery voltage, and abnormal sensor behavior.

## Project Overview

Modern vehicles use many sensors to monitor system behavior and detect possible problems. This project creates a small simulated automotive dataset and trains a machine learning model to classify different vehicle operating conditions.

The project includes:

- Simulated automotive sensor data
- Data preprocessing with pandas
- Machine learning classification using scikit-learn
- Fault detection with a Random Forest model
- Model evaluation using a classification report
- Confusion matrix visualization
- Feature importance analysis
- Automated output report generation

## Fault Classes

The model classifies the sensor data into the following classes:

- Normal operation
- High temperature fault
- Low battery fault
- Sensor anomaly

## Sensor Data Used

The dataset includes the following simulated vehicle sensor values:

- Vehicle speed
- Engine RPM
- Battery voltage
- Motor temperature
- Brake pressure
- Throttle position
- Vibration level

## Technologies Used

- Python
- pandas
- scikit-learn
- matplotlib
- Random Forest Classifier

## Project Structure

```text
automotive-sensor-fault-detection-ml/
│
├── data/
│   └── vehicle_sensor_data.csv
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── ml_report.txt
│
├── automotive_fault_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to Run

Clone the repository:
''' https://github.com/vaishnavijoshi438-oss/Automotive-Sensor-Fault-Detection-with-Machine-Learning.git
'''

Open the project folder:

```bash
cd automotive-sensor-fault-detection-ml
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python automotive_fault_detection.py
```

## Output Files

After running the script, the following files are created:

```text
data/vehicle_sensor_data.csv
outputs/confusion_matrix.png
outputs/feature_importance.png
outputs/ml_report.txt
```

## Example Results

The project generates:

- A CSV file containing simulated automotive sensor data
- A confusion matrix showing model classification performance
- A feature importance chart showing which sensor values influence the prediction
- A text report with precision, recall and F1-score values

## Skills Demonstrated

- Python programming
- Machine learning basics
- Data analysis with pandas
- Classification using scikit-learn
- Automotive sensor data interpretation
- Fault detection logic
- Data visualization with matplotlib
- Automated report generation
- Clean project structure

## Future Improvements

- Use a larger dataset
- Add real vehicle or CAN bus data
- Add more fault classes
- Compare multiple machine learning models
- Add anomaly detection methods
- Build a dashboard for sensor monitoring
- Add live sensor data simulation

