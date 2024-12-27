# Project README

This repository contains code, logs, and various configuration files for a TensorFlow-based trading bot. Below is a high-level overview of each file and directory, along with usage instructions.

---

## 1. Logs Directory

The `logs` folder contains timestamped log files from runs of the trading bot. Each file typically records information about:

- The training process (such as the number of epochs, data shape, model summary).
- Any warnings or errors encountered (for example, dimension mismatches in the LSTM layers).

Example log files:

- `trading_bot_20241222_211414.log`
- `trading_bot_20241222_212435.log` 
- `trading_bot_20241222_212049.log`
- ... (and so on)

These files are mostly for debugging. If a run fails with an error, searching these log files can help you trace the cause (for instance, the "Dimensions must be equal" error for an LSTM layer input mismatch).

---

## 2. requirements.txt

This file lists the Python dependencies needed to run or develop the trading bot. For example:

- `numpy>=1.26.0,<2.0.0`
- `tensorflow>=2.6.0,<2.19.0`
- `pandas>=1.3.0`
- ... etc.

To install everything in a fresh environment, you would typically run:

```
pip install -r requirements.txt
```

Make sure you have Python 3.9+ (as TensorFlow 2.18 requires Python ≥ 3.9).

---

## 3. trading_env/Lib/site-packages/tensorflow/...

Within this directory, you'll find files and folders that belong to the TensorFlow package installation. Typically, you do not need to modify these files directly; they are part of the underlying TensorFlow library. However, they're included here if you need to inspect TensorFlow's internal behavior for debugging:

- `python/pywrap_tensorflow.py`
  - Loads the native TensorFlow runtime and provides error messages if it fails.
- `tools/compatibility/tf_upgrade_v2.py`  
  - A script that helps migrate TensorFlow 1.x code to 2.x.
  - You generally wouldn't manually edit this in a normal workflow; it's provided by TensorFlow for backward compatibility.
- `tools/pip_package/setup.py`
  - The setup script for building the official TensorFlow pip package.
- `api/v2/...`
  - Various submodules under the TensorFlow v2 API namespace.

### Why these files appear

When the trading bot is installed in a virtual environment (or a site-packages directory), the TensorFlow library spreads out its internal modules. They show up here if you're browsing the environment.

---

## 4. trading_env/Lib/site-packages/tensorflow-2.18.0.dist-info/...

These files store "metadata" about the TensorFlow 2.18.0 installation. Examples:

- `METADATA`
  - High-level information about the TensorFlow package: name, version, author, dependencies required, etc.
- `REQUESTED`
  - Indicates that TensorFlow was explicitly installed (as opposed to being pulled in as a dependency).
- `WHEEL` / `RECORD`
  - Technical metadata used by pip to track installed files, versions, checksums, etc.

Likewise for:

`trading_env/Lib/site-packages/tensorflow_intel-2.18.0.dist-info/...`

This is another distribution-info folder that represents the CPU/GPU-accelerated Intel build of TensorFlow 2.18.0.

---

## 5. High-Level Usage

Below is a typical workflow for using this trading bot (assuming it's set up in a Python environment):

### Activate Your Virtual Environment

Windows:

```
python -m venv venv
.\venv\Scripts\activate
```

Mac/Linux:

```
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the Trading Bot

Depending on how the main script is set up, you might do something like:

```
python main.py
```

(or whichever file boots the training and inference process)

### Monitor Logs

If the bot encounters errors, or if you want to check the training progress, refer to the `logs/*.log` files. They contain detailed debugging info such as dimension mismatches in LSTM layers or other stack traces.

---

## 6. Modifying or Extending the Code

- If you need to make changes to the model architecture (for example, adjusting LSTM layers or the number of features), you would typically look for Python files in a subfolder named `models` or in a main script.
- You can also customize training hyperparameters (such as the number of epochs or batch size) in the script that creates and trains your model.
- If you need to add or remove columns from your data, you'll likely adjust your preprocessing script or the data loading logic.

---

## 7. Troubleshooting

- If you see dimension mismatch errors in the logs (e.g., "Dimensions must be equal, but are 12 and 9"), that usually indicates an inconsistency between the shape of your input data and the LSTM layer's expected input dimension.
- Check the logs for more details, and confirm that your data shape `(batch_size, timesteps, features)` matches the `input_shape` specified in your LSTM layer(s).

---

## 8. Summary

This project bundles a TensorFlow-based trading bot, related logs, and environment specifications. Here's a concise rundown of where to look for common tasks:

- `logs`: Check these to see any training or runtime errors.
- `requirements.txt`: Install your Python dependencies.
- `trading_env/Lib/site-packages/tensorflow/`: TensorFlow code and info. Typically do not modify these unless you're debugging deep TensorFlow internals.
- `.dist-info` directories: Pipeline metadata about installed packages.
