import os
import pandas as pd
import matplotlib.pyplot as plt

STAT_FOLDER = "./statistics"
OUT_FOLDER = "./graphs/"
def process_csv(csv_path):

    df = pd.read_csv(csv_path)
    # Metrics
    df["mean_score"] = df["score"].rolling(window=10, min_periods=1).mean()

    df["reward_mean"] = df["accumulated_reward"].rolling(window=10, min_periods=1).mean()
    df["reward_std"]  = df["accumulated_reward"].rolling(window=10, min_periods=1).std()

    threshold = 1.5
    df["reward_crash"] = df["accumulated_reward"] < (
        df["reward_mean"] - threshold * df["reward_std"]
    )

    highscore = df["score"].max()
    overall_mean_score = df["score"].mean()

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(df["episode"], df["score"], label="Score per Episode")
    plt.plot(df["episode"], df["mean_score"], label="Mean Score (10 eps)", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Score & Mean Score")
    plt.legend()

    text_x = df["episode"].max() * 0.88
    text_y = df["score"].max() * 0.9

    plt.text(
        text_x,
        text_y,
        f"Highscore: {highscore}\nMean Score: {overall_mean_score:.2f}",
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
    )

    plt.subplot(2, 1, 2)
    plt.plot(df["episode"], df["accumulated_reward"], label="Accumulated Reward")
    plt.plot(df["episode"], df["reward_mean"], linewidth=2, label="Reward Mean")

    plt.scatter(
        df[df["reward_crash"]]["episode"],
        df[df["reward_crash"]]["accumulated_reward"],
        color="red",
        s=30,
        label="Reward Crash"
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Accumulated Reward")
    plt.legend()

    # SAVE
    plt.tight_layout()

    base = os.path.basename(csv_path).replace(".csv", "").replace("statistics_","")

    os.makedirs(OUT_FOLDER, exist_ok = True)
    out_path = f"./graphs/training_analysis_{base}.png"

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved {out_path}")


for file in os.listdir(STAT_FOLDER):
    if file.lower().endswith(".csv"):
        full_path = os.path.join(STAT_FOLDER, file)
        print(f"Processing {file} ...")
        process_csv(full_path)

print("Done generating images")
