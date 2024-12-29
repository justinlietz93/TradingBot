import sys
import logging
from main import main

if __name__ == "__main__":
    try:
        sys.argv = ["main.py", "--mode", "backtest", "--model_path", "models/checkpoints/20241228-195417/final_model.keras"]
        main()
    except Exception as e:
        logging.error(f"Error running backtest: {str(e)}")
        sys.exit(1) 