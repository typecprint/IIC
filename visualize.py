import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from torchvision.utils import make_grid
from model import ModelIIC
from dataset import create_dataloaders
import matplotlib

matplotlib.use("Agg")  # 保存用にバックエンドを設定


def visualize_results(model_path="./model/resnet/model.pt", data_dir="./data"):
    print("Loading data and model for visualization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DataLoaderの取得 (Testデータのみ使用)
    _, test_loader = create_dataloaders(data_dir, batch_size=256)

    # モデルのロード
    model = ModelIIC(num_classes=10)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}. Using untrained model.")

    model.to(device)
    model.eval()

    pred_list = []
    true_list = []
    images = []  # 可視化用画像

    with torch.no_grad():
        for x, _, label in test_loader:
            x = x.to(device)
            phi_x = model(x)
            _pred = torch.argmax(phi_x, dim=1)

            pred_list.extend(_pred.cpu().numpy())
            true_list.extend(label.numpy())
            images.append(x.cpu())

    images = torch.cat(images)
    pred_list = np.array(pred_list)
    true_list = np.array(true_list)

    # 1. 混同行列(クラスタの分布)のヒートマップ作成
    matrix = np.zeros((10, 10))
    for i in range(len(pred_list)):
        row = true_list[i] if true_list[i] < 10 else 9
        col = pred_list[i]
        matrix[row][col] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted Cluster")
    plt.ylabel("True Class")
    plt.title("Cluster Assignment Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()
    print("Saved confusion matrix heatmap as 'confusion_matrix.png'")

    # 2. 各クラスタの代表画像を表示
    num_clusters = 10
    num_samples_per_cluster = 8

    fig, axes = plt.subplots(num_clusters, 1, figsize=(10, 2 * num_clusters))

    for cluster_id in range(num_clusters):
        # このクラスタに割り当てられた画像のインデックスを取得
        cluster_indices = np.where(pred_list == cluster_id)[0]

        if len(cluster_indices) > 0:
            # ランダムにサンプルを選択
            sample_indices = np.random.choice(
                cluster_indices,
                min(num_samples_per_cluster, len(cluster_indices)),
                replace=False,
            )
            sample_images = images[sample_indices]

            # 画像のテンソル(B, C, H, W)を標準化から元に戻す (近似)
            # STL-10の標準的な正規化が使われていると仮定
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            sample_images = sample_images * std + mean
            sample_images = torch.clamp(sample_images, 0, 1)

            # グリッド画像の作成
            grid = make_grid(
                sample_images, nrow=num_samples_per_cluster, normalize=False
            )

            axes[cluster_id].imshow(np.transpose(grid.numpy(), (1, 2, 0)))
            axes[cluster_id].set_title(
                f"Cluster {cluster_id} Examples (Count: {len(cluster_indices)})"
            )
            axes[cluster_id].axis("off")
        else:
            axes[cluster_id].text(
                0.5, 0.5, "No images in this cluster", ha="center", va="center"
            )
            axes[cluster_id].set_title(f"Cluster {cluster_id}")
            axes[cluster_id].axis("off")

    plt.tight_layout()
    plt.savefig("cluster_examples.png")
    plt.close()
    print("Saved cluster example images as 'cluster_examples.png'")


if __name__ == "__main__":
    visualize_results()
