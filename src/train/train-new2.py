# autopep8: off
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
# autopep8: on

import argparse
import datetime
import inspect
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, Image
import glob
import time

tf.keras.backend.set_floatx("float64")

from weighting import weighting
import levenberg_marquardt as lm

# ==============================
# PlotLossCallback: 仅保存图片到文件（无 Jupyter 依赖）
# ==============================
# class PlotLossCallback(Callback):
#     def __init__(self, workdir, figsize=(12, 5), plot_interval=5):
#         self.figsize = figsize
#         self.workdir = workdir
#         self.plot_interval = plot_interval
#         self.train_losses = []
#         self.val_losses = []
#         self.train_wmse = []
#         self.val_wmse = []
    
#     def on_epoch_end(self, epoch, logs=None):
#         # 收集指标
#         self.train_losses.append(logs.get('loss', 0))
#         self.val_losses.append(logs.get('val_loss', 0))
#         self.train_wmse.append(logs.get('weighted_mean_squared_error', 0))
#         self.val_wmse.append(logs.get('val_weighted_mean_squared_error', 0))
        
#         # 每 plot_interval 个 epoch 保存一次曲线图
#         if (epoch + 1) % self.plot_interval == 0:
#             # 防止 log(0) 或负值
#             train_losses = np.clip(self.train_losses, 1e-10, None)
#             val_losses = np.clip(self.val_losses, 1e-10, None)
#             train_wmse = np.clip(self.train_wmse, 1e-10, None)
#             val_wmse = np.clip(self.val_wmse, 1e-10, None)
            
#             # 创建双图
#             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
            
#             # MSE 曲线
#             ax1.semilogy(train_losses, label='Train MSE', linewidth=2)
#             ax1.semilogy(val_losses, label='Val MSE', linewidth=2)
#             ax1.set_xlabel('Epoch')
#             ax1.set_ylabel('Mean Squared Error (log scale)')
#             ax1.set_title(f'Training MSE - Epoch {epoch+1}')
#             ax1.legend()
#             ax1.grid(True, which="both", ls="-")
            
#             # Weighted MSE 曲线
#             ax2.semilogy(train_wmse, label='Train Weighted MSE', linewidth=2)
#             ax2.semilogy(val_wmse, label='Val Weighted MSE', linewidth=2)
#             ax2.set_xlabel('Epoch')
#             ax2.set_ylabel('Weighted MSE (log scale)')
#             ax2.set_title(f'Training Weighted MSE - Epoch {epoch+1}')
#             ax2.legend()
#             ax2.grid(True, which="both", ls="-")
            
#             # 保存并关闭
#             plt.tight_layout()
#             plt.savefig(
#                 os.path.join(self.workdir, f'loss_curve_epoch_{epoch+1:04d}.png'),
#                 dpi=150,
#                 bbox_inches='tight'
#             )
#             plt.close(fig)  # 释放内存，防止内存泄漏

#     def on_train_end(self, logs=None):
#         """训练结束时保存最终完整曲线图 + 历史数据"""
#         # 保存最终完整曲线
#         train_losses = np.clip(self.train_losses, 1e-10, None)
#         val_losses = np.clip(self.val_losses, 1e-10, None)
#         train_wmse = np.clip(self.train_wmse, 1e-10, None)
#         val_wmse = np.clip(self.val_wmse, 1e-10, None)
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
#         ax1.semilogy(train_losses, label='Train MSE', linewidth=2)
#         ax1.semilogy(val_losses, label='Val MSE', linewidth=2)
#         ax1.set_xlabel('Epoch')
#         ax1.set_ylabel('Mean Squared Error (log scale)')
#         ax1.set_title('Final Training MSE')
#         ax1.legend()
#         ax1.grid(True, which="both", ls="-")
        
#         ax2.semilogy(train_wmse, label='Train Weighted MSE', linewidth=2)
#         ax2.semilogy(val_wmse, label='Val Weighted MSE', linewidth=2)
#         ax2.set_xlabel('Epoch')
#         ax2.set_ylabel('Weighted MSE (log scale)')
#         ax2.set_title('Final Training Weighted MSE')
#         ax2.legend()
#         ax2.grid(True, which="both", ls="-")
        
#         plt.tight_layout()
#         plt.savefig(
#             os.path.join(self.workdir, 'final_loss_curve.png'),
#             dpi=150,
#             bbox_inches='tight'
#         )
#         plt.close(fig)
        
#         # 保存历史数据到 JSON（方便后续分析）
#         history = {
#             "train_loss": self.train_losses,
#             "val_loss": self.val_losses,
#             "train_weighted_mse": self.train_wmse,
#             "val_weighted_mse": self.val_wmse
#         }
#         with open(os.path.join(self.workdir, "training_history.json"), "w") as f:
#             json.dump(history, f, indent=2)

class PlotLossCallback(Callback):
    def __init__(self, workdir, figsize=(18, 5), plot_interval=5):
        self.figsize = figsize
        self.workdir = workdir
        self.plot_interval = plot_interval
        self.train_losses = []
        self.val_losses = []
        self.train_mse = []
        self.val_mse = []
        self.train_wmse = []
        self.val_wmse = []

    def on_epoch_end(self, epoch, logs=None):
        # 收集指标
        self.train_losses.append(logs.get('loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.train_mse.append(logs.get('mean_squared_error', 0))
        self.val_mse.append(logs.get('val_mean_squared_error', 0))
        self.train_wmse.append(logs.get('weighted_mean_squared_error', 0))
        self.val_wmse.append(logs.get('val_weighted_mean_squared_error', 0))

        # 每 plot_interval 个 epoch 保存一次曲线图
        if (epoch + 1) % self.plot_interval == 0:
            self._plot(epoch + 1, final=False)

    def on_train_end(self, logs=None):
        """训练结束时保存最终完整曲线图 + 历史数据"""
        self._plot(len(self.train_losses), final=True)

        # 保存历史数据到 JSON（方便后续分析）
        history = {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "train_mse": self.train_mse,
            "val_mse": self.val_mse,
            "train_weighted_mse": self.train_wmse,
            "val_weighted_mse": self.val_wmse,
        }
        with open(os.path.join(self.workdir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    def _plot(self, epoch, final=False):
        """绘制三联图"""
        # 防止 log(0) 或负值
        train_losses = np.clip(self.train_losses, 1e-10, None)
        val_losses = np.clip(self.val_losses, 1e-10, None)
        train_mse = np.clip(self.train_mse, 1e-10, None)
        val_mse = np.clip(self.val_mse, 1e-10, None)
        train_wmse = np.clip(self.train_wmse, 1e-10, None)
        val_wmse = np.clip(self.val_wmse, 1e-10, None)

        fig, axes = plt.subplots(1, 3, figsize=self.figsize)

        # Loss 曲线
        axes[0].semilogy(train_losses, label='Train Loss', linewidth=2)
        axes[0].semilogy(val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (log scale)')
        axes[0].set_title(f'Loss - Epoch {epoch}')
        axes[0].legend()
        axes[0].grid(True, which="both", ls="-")

        # MSE 曲线
        axes[1].semilogy(train_mse, label='Train MSE', linewidth=2)
        axes[1].semilogy(val_mse, label='Val MSE', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE (log scale)')
        axes[1].set_title(f'MSE - Epoch {epoch}')
        axes[1].legend()
        axes[1].grid(True, which="both", ls="-")

        # Weighted MSE 曲线
        axes[2].semilogy(train_wmse, label='Train Weighted MSE', linewidth=2)
        axes[2].semilogy(val_wmse, label='Val Weighted MSE', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Weighted MSE (log scale)')
        axes[2].set_title(f'Weighted MSE - Epoch {epoch}')
        axes[2].legend()
        axes[2].grid(True, which="both", ls="-")

        plt.tight_layout()
        fname = "final_loss_curve.png" if final else f"loss_curve_epoch_{epoch:04d}.png"
        plt.savefig(os.path.join(self.workdir, fname), dpi=150, bbox_inches='tight')
        plt.close(fig)

# ==============================
# 原有训练逻辑（精简打印）
# ==============================
HEADER = r"""
                      _____ ____  _____  ______  _____                          
                     / ____/ __ \|  __ \|  ____|/ ____|                         
                    | |   | |  | | |__) | |__  | (___                           
                    | |   | |  | |  ___/|  __|  \___ \                          
                    | |___| |__| | |    |____ ____) |                         
                     \_____\___\_\_|    |______|_____/                          
"""

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="prepare.json", type=str)
    args = parser.parse_args()
    return args

def _check_config(config):
    if not os.path.exists(config["data"]):
        raise FileNotFoundError(f"Data directory {config['data']} not found.")

def _print_header():
    print(f"{HEADER:^80}")
    print(f"{'CQPES: ChongQing Potential Energy Surface (legacy)':^80}")
    print(f"{'TRAIN':^80}")
    print("=" * 80)

if __name__ == "__main__":
    _print_header()
    args = _parse_args()

    with open(args.config) as f:
        config = json.load(f)
    _check_config(config)

    # ========== 数据加载 ==========
    print(f"Loading data from: {os.path.abspath(config['data'])}")
    X = np.load(os.path.join(config["data"], "X.npy"))[:, 1:]
    y = np.load(os.path.join(config["data"], "y.npy"))
    V = np.load(os.path.join(config["data"], "V.npy"))

    # ========== 数据分割 ==========
    index = np.arange(len(X))
    train_idx, valid_idx = train_test_split(index, test_size=(1 - config['split'][0]))
    valid_idx, test_idx = train_test_split(valid_idx, test_size=(config['split'][2] / (1 - config['split'][0])))

    workdir = os.path.abspath(f"{config['workdir']}-{datetime.datetime.now()}")
    os.makedirs(workdir, exist_ok=True)

    # 保存索引
    np.savetxt(os.path.join(workdir, "train_idx.txt"), train_idx, fmt="%d")
    np.savetxt(os.path.join(workdir, "valid_idx.txt"), valid_idx, fmt="%d")
    np.savetxt(os.path.join(workdir, "test_idx.txt"), test_idx, fmt="%d")

    # ========== 模型构建 ==========
    weights = np.array([weighting(v) for v in V])
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(len(X[0]),)))
    for num_units in config["network"]["layers"]:
        model.add(tf.keras.layers.Dense(units=num_units, activation=config["network"]["activation"]))
    model.add(tf.keras.layers.Dense(units=1, activation="linear"))

    # ========== 编译模型 ==========
    model_wrapper = lm.ModelWrapper(model)
    model_wrapper.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=config['fit']['lr']),
        loss=lm.MeanSquaredError(),
        damping_algorithm=lm.DampingAlgorithm(
            adaptive_scaling=config['lm']['adaptive_scaling'],
            fletcher=config['lm']['fletcher'],
        ),
        solve_method=config['lm']['solve_method'],
        jacobian_max_num_rows=config['lm']['jacobian_max_num_rows'],
        metrics=[tf.keras.metrics.MeanSquaredError()],
        weighted_metrics=[tf.keras.metrics.MeanSquaredError()],
    )

    # ========== 回调设置 ==========
    ckpt_dir = os.path.join(workdir, "ckpt")
    log_dir = os.path.join(workdir, "log")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "model-epoch-{epoch:04d}-val-mse-{val_weighted_mean_squared_error:.5e}.h5"),
            monitor="val_weighted_mean_squared_error",
            save_best_only=True,
            save_weights_only=True,
        ),
        TensorBoard(log_dir),
        PlotLossCallback(workdir=workdir, figsize=(12, 5), plot_interval=5),  # 每5个epoch保存一次图
        EarlyStopping(monitor="val_loss", patience=100, restore_best_weights=True),
    ]

    # ========== 开始训练 ==========
    batch_size = len(train_idx) if config['fit']['batch_size'] == -1 else config['fit']['batch_size']
    
    print(f"\n🚀 Starting training with {len(train_idx)} samples...")
    print(f"   Epochs: {config['fit']['epoch']} | Batch Size: {batch_size}")
    print(f"   Workdir: {workdir}\n")
    
    # 启动训练
    model_wrapper.fit(
        X[train_idx],
        y[train_idx],
        batch_size=batch_size,
        epochs=config['fit']['epoch'],
        sample_weight=weights[train_idx],
        validation_data=(X[valid_idx], y[valid_idx], weights[valid_idx]),
        callbacks=callbacks,
        verbose=0  # 关闭 Keras 默认进度条
    )
    
    print(f"\n✅ Training completed! Results saved in: {workdir}")
    print(f"   - Final loss curve: {os.path.join(workdir, 'final_loss_curve.png')}")
    print(f"   - Training history: {os.path.join(workdir, 'training_history.json')}")
    print(f"   - Best model: {ckpt_dir}")
