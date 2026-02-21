import torch
import numpy as np
import logging
import os
import time

from dataset import create_dataloaders
from model import ModelIIC, IID_loss


# 念入りなデバッグのためのロガー設定
def setup_logger(log_file="iic_training.log"):
    logger = logging.getLogger("IIC_Logger")
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # コンソール出力 (標準出力)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # ファイル出力
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


logger = setup_logger()


def train(epoch, model, train_loader, optimizer, device):
    logger.info(f"--- Epoch {epoch} Training Started ---")
    model.train()

    n_batch = len(train_loader)
    loss_ep = []

    start_time = time.time()

    for i, (x, x_prime, _) in enumerate(train_loader, start=1):
        x = x.to(device)
        x_prime = x_prime.to(device)

        optimizer.zero_grad()

        # Generate representations
        phi_x = model(x)
        phi_x_prime = model(x_prime)

        # Calculate Mutual Information (IIC loss)
        loss = IID_loss(phi_x, phi_x_prime)
        loss_ep.append(loss.item())

        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == n_batch:
            avg_loss = (
                np.mean(loss_ep[-10:]) if len(loss_ep) >= 10 else np.mean(loss_ep)
            )
            logger.debug(
                f"[Train] Epoch {epoch} | Batch {i}/{n_batch} | Current Loss: {loss.item():.5f} | Avg 10-batch Loss: {avg_loss:.5f}"
            )

    end_time = time.time()
    epoch_avg_loss = np.mean(loss_ep)
    logger.info(
        f"--- Epoch {epoch} Completed | Avg Loss: {epoch_avg_loss:.5f} | Time: {end_time - start_time:.2f}s ---"
    )

    return model, optimizer


def test(model, test_loader, device):
    logger.info("--- Testing Started ---")
    model.eval()

    pred_list = []
    true_list = []
    n_batch = len(test_loader)

    start_time = time.time()

    with torch.no_grad():
        for i, (x, _, label) in enumerate(test_loader, start=1):
            x = x.to(device)

            # Generate cluster prediction
            phi_x = model(x)
            _pred = torch.argmax(phi_x, dim=1)

            pred_list.extend(_pred.cpu().numpy())
            true_list.extend(label.numpy())

            if i % 10 == 0 or i == n_batch:
                logger.debug(f"[Test] Processing Batch {i}/{n_batch}")

    end_time = time.time()
    logger.info(f"--- Testing Completed | Time: {end_time - start_time:.2f}s ---")

    return np.array(pred_list), np.array(true_list)


def main():
    logger.info("Starting IIC Process.")

    # 1. パラメータの設定
    dataset_dir = "./data"  # 作業ディレクトリ下にDLするよう再設定
    batch_size = 256
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. データ準備
    logger.info("Preparing STL-10 dataset (Download and Loader creation)...")
    try:
        train_loader, test_loader = create_dataloaders(
            dataset_dir, batch_size=batch_size
        )
    except Exception as e:
        logger.error(f"Failed to prepare dataloaders: {e}", exc_info=True)
        return

    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # 3. モデル・オプティマイザ設定
    logger.info("Initializing Model and Optimizer...")
    model = ModelIIC(num_classes=10)
    model.to(device)

    # Classification layerのみ学習可能 (sandbox.pyと同様)
    optimizer = torch.optim.Adam(model.model_conv.classifier.parameters(), lr=1e-4)
    logger.debug(f"Optimizer set with lr=1e-4 on model classifier.")

    # 4. 学習ループ
    for epoch in range(1, epochs + 1):
        model, optimizer = train(epoch, model, train_loader, optimizer, device)

    # 5. テスト評価
    pred, true = test(model, test_loader, device)

    # Note: STL-10 labels are 0-indexed internally in torchvision,
    # but depending on evaluation we might need offset.
    # Here, we keep torchvision's format.

    # 6. 混同行列(Confusion Matrix)の作成 (Cluster assignments)
    # クラスターの割り当てはクラスラベルと完全一致しない(クラスタリングであるため)
    # しかし行列の分布を確認する
    matrix = np.zeros((10, 10))
    for i in range(len(pred)):
        # Ensure indices stay within bounds
        row = true[i] if true[i] < 10 else 9
        col = pred[i]
        matrix[row][col] += 1

    np.set_printoptions(suppress=True)
    logger.info("Cluster assignment matrix against true labels:")
    logger.info("\n" + str(matrix))

    # 7. モデルの保存
    os.makedirs("./model/resnet", exist_ok=True)
    save_path = "./model/resnet/model.pt"
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    logger.info("Finished IIC Process.")


if __name__ == "__main__":
    main()
