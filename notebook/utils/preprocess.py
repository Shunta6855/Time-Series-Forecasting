# ライブラリのインポート
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# DataWindowクラスを定義
class DataWindow():
    """
    時系列データのウィンドウ化とデータセット管理を行うクラス

    Args:
        input_width (int): 入力データのウィンドウサイズ
        label_width (int): ラベルデータのウィンドウサイズ
        shift (int): 入力からラベルへのシフト量
        train_df (pd.DataFrame): 訓練データ
        val_df (pd.DataFrame): 検証データ
        test_df (pd.DataFrame): テストデータ
        label_columns (list[str], optional): ラベルとする列の名前。デフォルトはNone

    Attributes:
        input_width (int): 入力データのウィンドウサイズ
        label_width (int): ラベルデータのウィンドウサイズ
        shift (int): 入力からラベルへのシフト量
        total_window_size (int): 入力とラベルを合わせたウィンドウサイズ
        columns_indices (dict): 各列の名前とインデックス
        label_columns_indices (dict): ラベル列の名前とインデックス
    """
    # DataWindowクラスの初期化メソッド（インスタンスを作成した際に自動で実行される処理）を定義
    # 基本的に変数を割り当て、入力とラベルのインデックスを管理する
    def __init__(self, input_width: int, label_width: int, shift: int, 
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                 label_columns: list[str] = None):

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns # 予測したいラベル列の名前
        if label_columns is not None:
            # この列の名前とインデックスからdictを作成（プロットに使う）
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)
            }
        # 各列の名前とインデックスからdictを作成
        # 目的変数と特徴量を分けるために使う
        self.columns_indices = {
            name: i for i, name in enumerate(train_df.columns)
        }

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift

        # slice関数はシーケンスのスライス方法を指定するスライスオブジェクトを返す
        # この場合、入力スライスは0から始まり、input_widthに達した時点で終わる
        self.input_slice = slice(0, input_width) # indexを返す
        # 入力にインデックスを割り当てる（プロットに役立つ）
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
            # arange関数は指定された範囲と刻み幅に基づいて等差数列を生成する
        
        # ラベルが始まる位置のインデックスを取得
        # この場合は、ウィンドウ全体のサイズからラベルの幅を引いたものになる
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        # 入力と同じ手順をラベルにも適用
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
    
    # ウィンドウを入力とラベルに分割（これにより、モデルが入力に基づいて予測値を生成し、ラベルに対して誤差指標を計測できるようにする）
    def split_to_inputs_labels(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ウィンドウを入力とラベルに分割

        Args:
            features (torch.Tensor): 入力特徴量

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 入力データとラベルデータ
        """
        # __init__で定義したinput_sliceを使ってウィンドウをスライスし、入力を取得
        inputs = features[self.input_slice, :]

        # __init__で定義したlabel_sliceを使ってウィンドウをスライスし、ラベルを取得
        labels = features[self.labels_slice, :]

        # 目的変数が複数の場合は、ラベルを積み重ねる
        if self.label_columns is not None:
            labels = torch.stack(
                [labels[:, self.columns_indices[name]] # この段階では、目的変数に対応する列データを1つずつlabelsから抜き出していることから、得られるのは(バッチサイズ、時系列長)の2次元テンソル
                 for name in self.label_columns],
                 dim=-1)
                # 最後にtorch.stackによって作成された1次元テンソルを結合して2次元テンソルを作成している
                    # dim=-1によって新しい次元を最後の次元（特徴量次元）として追加する
        return inputs, labels

    # データウィンドウのサンプルをプロットするメソッド
    def plot(self, model=None, plot_col='traffic_volume', max_subplots=3):
        inputs, labels = self.sample_batch

        plt.figure(figsize=(12, 8))
        plot_col_index = self.columns_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        for n in range(max_n):
            plt.subplot(3, 1, n+1) # n+1によってプロットするfigを指定
            plt.ylabel(f'{plot_col}[scaled]')
            # 入力をプロット：ドット付きの青の連続する線で表される
            plt.plot(self.input_indices, inputs[n, :, plot_col_index].detach().numpy(), 
                        # バッチ内のn番目のデータのplot_col_indexに対応する列をプロット
                        # detach()により計算グラフから切り離し、numpy()によってPyTorchのテンソルをNumPy配列に変換
                     label='Inputs', marker='.', zorder=-10)
                        # zorder=-10：背景レイヤーでプロット（他の要素より後ろに描画）
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index].detach().numpy(),
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index].detach().numpy(),
                            marker='X', edgecolors='k', label='Predictions',
                            c='red', s=64)
            if n == 0:
                plt.legend()

        plt.xlabel('Time(h)')
    
    # PyTorchでの学習を行うためにデータをinputとlabelに分け、さらにバッチサイズに分けるクラスを定義
    def make_dataset(self, data: pd.DataFrame) -> DataLoader:
        """
        指定されたデータフレームからDataLoaderを作成

        Args:
            data (pd.DataFrame): データフレーム

        Returns:
            DataLoader: バッチ処理可能なデータセット
        """
        data = np.array(data, dtype=np.float32)
        features = []

        for i in range(len(data)-self.total_window_size+1):
            window = data[i:i+self.total_window_size]
            features.append(window)

        features = np.stack(features) # リストfeaturesに含まれるすべてのウィンドウを1つの配列にまとめる
            # 結果の形状：(ウィンドウ数, ウィンドウ長, 特徴量数)
        dataset = TimeSeriesDataset(features, self.split_to_inputs_labels)
            # __getitem__メソッドにより、dataset[0]のようにインデックスを指定することで、その対象のウィンドウを取得できる
        return DataLoader(dataset, batch_size=32, shuffle=True)
            # DataLoader：PyTorchでデータをバッチ単位に分割し、シャッフルして取り出すためのクラス
            # →ここでバッチサイズが0次元目に加えられることから、dataset[0]の段階では2次元(ウィンドウ長(時系列長), 特徴量数)となるようにする
    
    @property
    def train(self) -> DataLoader:
        return self.make_dataset(self.train_df)
    
    @property
    def val(self) -> DataLoader:
        return self.make_dataset(self.val_df)
    
    @property
    def test(self) -> DataLoader:
        return self.make_dataset(self.test_df)
    
    @property
    def sample_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        サンプルバッチを取得する

        Returns:
            tuple[torch.Tensor, torch.Tensor]: サンプルの入力データとラベルデータ
        """
        result = getattr(self, '_sample_batch', None)
            # getattr: selfオブジェクト内で_sample_batch属性が存在するか確認する
        # resultがNoneの場合（初回アクセス時）、新たにデータを取得して_sample_batchに保存
        if result is None: 
            result = next(iter(self.train))
                # next: データローダーから最初のバッチを取り出す
            self._sample_batch = result
                # 取得したデータを_sample_batchという隠し属性に保存
                # →2回目以降のアクセス時にデータを再計算せず、キャッシュから直接データを返すことで効率化
        return result

# PyTorchのDatasetを継承した、時系列データを効率的に扱うためのカスタムデータセット
class TimeSeriesDataset(Dataset):
    """
    時系列データ用のカスタムデータセット。

    Args:
        features (np.ndarray): 特徴量データ
        split_fn (function): 入力とラベルを分割する関数
    """
    def __init__(self, features: np.ndarray, split_fn: callable):
        self.features = features
        self.split_fn = split_fn

    def __len__(self) -> int:
        """
        データセットの長さを返す(len())

        Returns:
            int: データセットのサンプル数
        """
        return len(self.features)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        インデックスに基づいてデータを取得します

        Args:
            idx (int): データセット内の特定のサンプル（ウィンドウ）のインデックス

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 入力データとラベルデータ
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return self.split_fn(features)
            # featuresテンソルの最初の次元（0番目）に新たな次元を追加→なし
                # split_fn関数は入力として(バッチサイズ、ウィンドウ長、特徴量数)の形状を期待している
                # →単一サンプルの次元(ウィンドウ長, 特徴量数)を(1, ウィンドウ長, 特徴量数)に変換
            # split_fn関数の入力として(ウィンドウ長, 特徴量数)を期待するように変更した
