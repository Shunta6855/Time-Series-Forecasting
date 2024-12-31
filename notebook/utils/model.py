import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple

class EarlyStopping:
    """
    検証lossが改善されなくなった場合にトレーニングを停止するクラス

    Args:
        patience(int): 改善が見られないエポック数の許容範囲
        delta(float): 改善とみなされる最小限の変化量
    """
    def __init__(self, patience: int = 3, delta: float = 0) -> None:
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        """
        検証lossの改善状況をチェック

        Args:
            val_loss(float): 現在の検証loss
        """
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class LinearModel(nn.Module):
    """
    線形回帰モデル

    Args:
        input_dim(int): 入力特徴量の次元数
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)# コンストラクタの引数を与える
            # 引数
                # 入力の特徴量の数
                # 出力の特徴量の数
                # 線形モデルにバイアス変数を用意するかどうかのフラグ（デフォルトでTrue）
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播計算

        Args:
            x(torch.Tensor): 入力テンソル

        Reuturns:
            torch.Tensor: 出力テンソル
        """
        return self.linear(x)
    
class DenseModel(nn.Module):
    """
    多層パーセプトロンモデル

    Args:
        input_dim(int): 入力特徴量の次元数
        output_dim(int): 出力特徴量の次元
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(DenseModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class LSTMModel(nn.Module):
    """
    LSTMモデル

    Args:
        input_dim(int): 入力特徴量の次元数
        output_dim(int): 出力特徴量の次元数
        hidden_dim(int): LSTMの隠れ層のユニット数
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int=32) -> None:
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            # batch_first=True: 入力テンソルの最初の次元がbatch_sizeであることを示す
                # デフォルトでは、LSTMの入力テンソルは次の順序を想定している
                    # (sequence_length, batch_size, input_size)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
            # _の部分には、(hn, cn)が渡される
                # hn: 最終の隠れ状態
                # cn: 最終のメモリセル
        return self.fc(x)

class CNNModel(nn.Module):
    """
    CNNモデル

    Args:
        input_dim(int): 入力特徴量の次元
        output_dim(int): 出力特徴量の次元
        kernel_size(int): 畳み込みカーネルのサイズ
    """
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int=3) ->None:
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
            # (バッチサイズ, 時系列長, 特徴量数)を入力としている
            # しかし、畳み込み演算は畳み込みの対象を最終次元に期待しているため、変換の必要がある
            # Kerasの Conv1D は、(batch_size, time_steps, features) の形式で入力されることを前提
            # → 次元順序の変換は不要
        x = self.conv1(x) 
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNNLSTMModel(nn.Module):
    """
    CNN + LSTMモデル

    Args:
        input_dim(int): 入力特徴量の次元数
        output_dim(int): 出力特徴量の次元数
        kernel_size(int): 畳み込みカーネルのサイズ
        hidden_dim(int): LSTM隠れ層のユニット層
    """
    def __init__(
            self, 
            input_dim: int, 
            output_dim: int, 
            kernel_size: int = 3, 
            hidden_dim: int = 32
    ) -> None:
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.lstm1 = nn.LSTM(32, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1) 
            # (バッチサイズ, 時系列長, 特徴量数) → (バッチサイズ, 特徴量数, 時系列長)
        x = self.conv1(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
            # (バッチサイズ, 特徴量数, 時系列長) → (バッチサイズ, 時系列長, 特徴量数)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return self.fc(x)
    
class AutoRegressive(nn.Module):
    """
    LSTMを用いたAutoRegressiveモデル

    Args:
        units(int): LSTMの隠れ層ユニット数
        out_steps(int): 出力ステップ数
        input_dim(int): 入力特徴量の次元
    """
    def __init__(self, units: int, out_steps: int, input_dim: int):
        super(AutoRegressive, self).__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = nn.LSTMCell(input_dim, units)
        self.dense = nn.Linear(units, input_dim)

    def warmup(self, inputs: torch.Tensor) -> tuple:
        """
        最初のステップでLSTMの初期状態と出力を取得する関数

        Args:
            inputs(torch.Tensor): 入力テンソルのシーケンス

        Returns:
            tuple: 初期予測値と状態
        """
        h, c = self.init_hidden_state(inputs.size(0))
            # inputs.size(0): 最初の次元のサイズ(batch_size)を取得
        x = inputs[:, -1, :] # 最後のタイムステップを使用
        h, c = self.lstm_cell(x, (h, c))
            # LSTMCellは1ステップごとに計算を行うセル単位のLSTM
            # → LSTMCellにおけるh(隠れ状態)はLSTMのoutputに相当する
        prediction = self.dense(h)
        return prediction, (h, c)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理

        Args:
            inputs(torch.Tensor): 入力テンソルのシーケンス

        Returns:
            torch.Tensor: 予測結果
        """
        predictions = []
        prediction, state = self.warmup(inputs)
        predictions.append(prediction)

        for _ in range(1, self.out_steps):
            x = prediction
            h, c = self.lstm_cell(x, state)
            prediction = self.dense(h)
            predictions.append(prediction)
            state = (h, c)

        predictions = torch.stack(predictions, dim=1)
            # 次元1に沿ってスタック(連結)することにより、次元1をout_stepsの次元にできる
        return predictions # (batch_size, out_steps, input_dim)

    def init_hidden_state(self, batch_size: int) -> tuple:
        """
        初期状態の生成

        Args:
            batch_size(int): バッチサイズ

        Returns:
            tuple: 隠れ状態(h)とセル状態(c)
        """
        h = torch.zeros(batch_size, self.units)
        c = torch.zeros(batch_size, self.units)
        return h, c

class ModelTrainer:
    """
    モデルのトレーニングと検証を実行するためのクラス
    
    Args:
        model(nn.Module): PyTorchモデル
        train_loader(DataLoader): 訓練データのDataLoader
        val_loader(DataLoader): 検証データのDataLoader
        criterion(nn.Module): 損失関数
        optimizer(optim.Optimizer): オプティマイザ
        device(torch.device): 実行環境(CPU/GPU)
    """
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            optimizer: optim.Optimizer,
            device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        
    def train(self, max_epochs: int = 50, patience: int = 3) -> Dict[str, List[float]]:
        """
        Early Stoppingを用いたモデルのトレーニング

        Args:
            max_epochs(int): 最大エポック数
            patience(int): 早期停止のエポック数

        Returns:
            Dict[str, List[float]]: トレーニング及び検証lossの履歴
        """
        early_stopping = EarlyStopping(patience=patience)
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(max_epochs):
            self.model.train()
            train_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /=len(self.train_loader)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in self.val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    val_outputs = self.model(x_val)
                    val_loss += self.criterion(val_outputs, y_val).item()
            val_loss /= len(self.val_loader)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            print(f'Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print('Early stopping triggered')
                break
        return history
