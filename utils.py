# utils.py

import joblib

def save_model(model, filename):
    # Save the trained model
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def save_plot(plot, filename):
    # Save the plot
    plot.savefig(filename)
    print(f"Plot saved as {filename}")
