import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_file", required=True)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    data = np.load(args.case_file)
    beta = data["beta"].squeeze()
    sensor_index = data["sensor_index"]
    np.set_printoptions(precision=12, suppress=False)

    print("beta shape:", beta.shape)
    print("beta min :", beta.min())
    print("beta max :", beta.max())
    print("beta mean:", beta.mean())
    print("beta std :", beta.std())
    print("unique beta count:", len(np.unique(beta)))
    print("first 20 beta:")
    print(beta[:20])

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.case_file)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.plot(sensor_index, beta)
    plt.xlabel("Sensor Index")
    plt.ylabel(r"Gate Value ($\beta$)")
    plt.title("Gate Value vs Sensor Index")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gate_value_vs_sensor_index.png"), dpi=300)  
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(beta, bins=30)
    plt.xlabel(r"Gate Value ($\beta$)")
    plt.ylabel("Count")
    plt.title("Gate Value Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gate_value_histogram.png"), dpi=300)        
    plt.close()
    order = np.argsort(beta)
    bottom_idx = order[:10]
    top_idx = order[-10:][::-1]

    pd.DataFrame({
        "sensor_idx": sensor_index[top_idx],
        "beta": beta[top_idx],
    }).to_csv(os.path.join(output_dir, "top10_high_gate_sensors.csv"), index=False)   

    pd.DataFrame({
        "sensor_idx": sensor_index[bottom_idx],
        "beta": beta[bottom_idx],
    }).to_csv(os.path.join(output_dir, "bottom10_low_gate_sensors.csv"), index=False) 

    print("Saved figures and CSV files to:", output_dir)


if __name__ == "__main__":
    main()