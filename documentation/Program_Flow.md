Program Flow Documentation

1. Program Entry Point
- The program starts execution from the main.py file.
- The main function in main.py is the entry point of the program.
- It loads the configuration settings from the config.py file.
- The main function initializes the MLModel class with the loaded configuration.


2. Data Loading and Preprocessing
- The data loading and preprocessing steps are handled in the data_loader.py and data_splitter.py files.
- The DataLoader class in data_loader.py is responsible for loading the raw data from external sources (e.g., CSV files or APIs).
- The loaded data is then preprocessed using various methods in the DataLoader class, such as cleaning, feature engineering, and normalization.
- The preprocessed data is split into training and testing sets using the split_data function in data_splitter.py.
- The split_data function takes the preprocessed data and splits it based on a specified ratio (e.g., 80% for training, 20% for testing).


3. Model Architecture
- The model architecture is defined in the ml_model.py file.
- The MLModel class in ml_model.py inherits from the BaseModel class defined in base_model.py.
- The build_model method in MLModel is responsible for constructing the LSTM model architecture.
- The LSTM model consists of multiple LSTM layers, followed by dropout, normalization, and dense layers.
- The _build_lstm_model method defines the specific architecture of the LSTM model, including the number of units, dropout rate, and activation functions.


4. Model Training
- The model training process is handled in the train method of the MLModel class.
- The train method takes the preprocessed training data (X_train and y_train) and validation data (X_val and y_val) as input.
- It reshapes the input data into the required 3D format (samples, timesteps, features) for LSTM input.
- The training process is configured with the specified number of epochs, batch size, and callbacks (e.g., early stopping, learning rate reduction).
- The model is compiled with an optimizer (e.g., Adam), loss function (e.g., custom_loss), and evaluation metrics (e.g., mean absolute error, direction accuracy).
- The model is trained using the fit method, which iterates over the training data for the specified number of epochs.
- The training progress is monitored using the validation data, and the best model weights are saved based on the validation loss.


5. Data Validation and Error Handling
- Data validation is performed in the validate_data method of the BaseModel class.
- The validate_data method checks the shape and format of the input data (X) and target data (y) to ensure they meet the expected requirements.
- Error handling is implemented throughout the codebase using try-except blocks to catch and handle exceptions gracefully.
- Logging statements are used to record important information, warnings, and errors during program execution.


6. Program Termination
- After the model training is completed, the program execution reaches the end of the main function in main.py.
- The trained model is saved to disk using the save_model method of the MLModel class.
- The program terminates, and the trained model can be used for making predictions on new data.


References:
- main.py: Program entry point and main function.
- data_loader.py: Data loading and preprocessing.
- data_splitter.py: Data splitting into training and testing sets.
- ml_model.py: Model architecture and training.
- base_model.py: Base class for models, including data validation and error handling.
