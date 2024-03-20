from os.path import isfile, basename
import sys
import glob
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
color_scheme = [
    "#347B98",  # Dark Slate Blue
    "#004165",  # Deep Navy
    "#183A37",  # Dark Slate Green
    "#5C4B51",  # Dark Mauve
    "#7F675B",  # Dark Beige
    "#4A3F45",  # Dark Purple
    "#5E503F",  # Umber
    "#3E2723",  # Deep Coffee
    "#1B4D3E",  # Brunswick Green
    "#4F4A45",  # Taupe
    "#423E37",  # Onyx
    "#3B3C36",  # Rifle Green
    "#343434",  # Jet
    "#2D4262",  # Dark Blue Gray
    "#1D2731",  # Charcoal
    "#0B3C5D",  # Dark Cerulean
]

def trial_post_proc(data_dir, name):

    loss_files = sorted(glob.glob(f"{data_dir}/*dqn_trainer_loss.csv"))
    if len(loss_files) == 1:
        loss_file = loss_files[0]
        df = pd.read_csv(loss_file, header=None, dtype=np.float32, on_bad_lines="skip")
        ep_data = np.array(df)
        plt.plot(ep_data)
        plt.title(f"{name} Trainer Loss")
        plt.xlabel("Step")
        plt.ylabel("Huber Loss")
        plt.savefig(f"{data_dir}/dqn_trainer_loss.png")
        plt.close()
    elif len(loss_files) == 16:
        fig = plt.figure()
        gs = fig.add_gridspec(4,4, hspace=0, wspace=0)
        axs = gs.subplots(sharex=True, sharey=True)
        #fig, axs = plt.subplots(4,4)
        fig.suptitle(f"{name} Trainer Loss")
        fig.supxlabel("Step")
        fig.supylabel("Huber Loss")
        col = 0
        for counter, loss_file in enumerate(loss_files):
            row = counter % 4
            if row == 0 and counter >= 3:
                col += 1
            subname = basename(loss_file)[:2]
            df = pd.read_csv(loss_file, header=None, dtype=np.float32, on_bad_lines="skip")
            ep_data = np.array(df)
            axs[row, col].plot(ep_data, color=color_scheme[counter], label="raw data")
            axs[row, col].set_title(subname, y=-0.01, color="orange")
            cumsum_vec = np.cumsum(np.insert(ep_data, 0, 0)) 
            window_width = 250
            ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
            axs[row, col].plot(ma_vec, color="red", label=f"{window_width} step sliding average")
        plt.legend()
        plt.savefig(f"{data_dir}/dqn_trainer_loss.png")
        plt.close()

    else:
        print(f"{loss_files} not found. Or != 16")

    csv_files = sorted(glob.glob(f"{data_dir}/infos/*.csv"), key=lambda x: int(x.split('_ep')[1].replace('.csv', '')))
    print(csv_files)
    delay = []
    total_stopped = []
    for csv in csv_files:
        df = pd.read_csv(csv)
        delay.extend(df["avg_delay"])
        total_stopped.extend(df["system_total_stopped"])

    plt.plot(delay, label="raw data")
    plt.title(f"{name} Average Imposed Delays")
    plt.xlabel("Episode")
    ticker = np.arange(0, len(delay), step=360*5)
    t_labels = np.arange(0, len(ticker))
    t_labels *= 5
    plt.xticks(ticker, t_labels)
    plt.ylabel("Delay (seconds)")
    cumsum_vec = np.cumsum(np.insert(delay, 0, 0)) 
    window_width = 250
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(ma_vec, color="red", label=f"{window_width} step sliding average")
    plt.legend()
    plt.savefig(f"{data_dir}/avg_delay.png")
    plt.close()

    plt.plot(total_stopped, label="raw data")
    plt.title(f"{name} System Total Stopped")
    plt.xlabel("Episode")
    ticker = np.arange(0, len(total_stopped), step=360*5)
    t_labels = np.arange(0, len(ticker))
    t_labels *= 5
    plt.xticks(ticker, t_labels)
    plt.ylabel("No. Vehicles")
    window_width = 250
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    plt.plot(ma_vec, color="red", label=f"{window_width} step sliding average")
    plt.legend()
    plt.savefig(f"{data_dir}/total_stopped.png")
    plt.close()


if __name__ == '__main__':
    trial_post_proc(sys.argv[1], "DQN")
