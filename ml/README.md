# Elderly Healthcare Monitoring - ML

This directory contains the machine learning pipeline for the wearable elderly healthcare monitoring system.

The current ML task is **fall detection** using time-series sensor data from a wearable belt.

## ML Responsibility

The ML component is responsible for:

1. Receiving accelerometer and gyroscope time-series data.
2. Processing sensor recordings into fixed-size windows.
3. Normalizing the sensor data using training-set statistics.
4. Detecting whether a window represents normal activity or a fall.
5. Producing a fall prediction that can later be passed to the AI/context layer.

The current model uses:

**1D CNN + LSTM**

The CNN learns short-term temporal patterns from the sensor signals, while the LSTM models temporal relationships across the resulting feature sequence.

---

## Dataset

The current prototype uses the **SisFall** dataset.

SisFall contains recordings from:

- 23 adult subjects (`SA01`-`SA23`)
- 15 elderly subjects (`SE01`-`SE15`)
- 38 subjects total
- Normal activities (`D01`-`D19`)
- Fall activities (`F01`-`F15`)

Each recording contains 9 sensor channels.

For the current fall-detection model, the first 6 channels are used:

| Channel | Sensor |
|---|---|
| 1 | ADXL345 acceleration X |
| 2 | ADXL345 acceleration Y |
| 3 | ADXL345 acceleration Z |
| 4 | ITG3200 rotation X |
| 5 | ITG3200 rotation Y |
| 6 | ITG3200 rotation Z |

The remaining MMA8451Q accelerometer channels are currently not used by the model.

The raw dataset is intentionally kept outside the Git repository.

---

## Classification Task

The first ML experiment is binary classification:

| Dataset code | ML label | Label ID |
|---|---|---:|
| `D01`-`D19` | Normal | 0 |
| `F01`-`F15` | Fall | 1 |

The model therefore produces two output classes:

```text
0 = Normal
1 = Fall
