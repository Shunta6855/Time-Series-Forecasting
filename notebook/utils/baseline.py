# ライブラリのインポート
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Union, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

# 入力データを予測値として返すクラス（PyTorchのnn.Moduleを継承）
class Baseline(nn.Module): 
    """
    入力データをそのまま予測値として返すベースラインモデル

    Args:
        label_index (int or list[int], optional): ラベルのインデックス。デフォルトはNone

    Attributes:
        label_index (int or list[int]): ラベルのインデックス
    """
    def __init__(self, label_index: Union[int, List[int]] = None):
        super(Baseline, self).__init__()
        self.label_index = label_index
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理(順伝播の処理をforwardに書く)

        Args:
            inputs (torch.Tensor): 入力データ

        Returns:
            torch.Tensor: 入力データから予測される出力
        """
                # →nn.Module内ではインスタンスを呼び出すと、同じ引数を持つforwardを呼び出すことになる
            # inputs: モデルに渡される入力データ
        # label_indexがNoneの場合、入力データをそのまま返す（デフォルト）
        if self.label_index is None:
            return inputs

        # label_indexがリストの場合、複数の特徴量（index）を抽出して、それらを結合して新しいテンソルを作る
        elif isinstance(self.label_index, list):
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                result = result.unsqueeze(-1) # 最後の次元に新しい次元を追加
                tensors.append(result)
            return torch.cat(tensors, dim=-1) #リストtensors内のテンソルを結合  
                # dim=-1により、最後の次元に沿って結合
        
        # label_indexが整数の場合、指定されたlabel_indexの特徴量を抽出し、次元を追加したテンソルを返す
        result = inputs[:, :, self.label_index]
        return result.unsqueeze(-1)

class MultiStepBaseline(nn.Module):
    """
    マルチステップ予測用のベースラインモデル

    Args:
        label_index(int): 予測対象のインデックス
    """
    def __init__(self, label_index: int = None):
        super(MultiStepBaseline, self).__init__()
        self.label_index = label_index

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理(次の24時間ステップにわたって目的変数列の最後の既知の値を返す)

        Args:
            inputs(torch.Tensor): 入力データ

        Returns:
            torch.Tensor: 最後のタイムステップを24回繰り返した出力
        """
        if self.label_index is None:
            # 目的変数が指定されない場合は、次の24時間ステップにわたってすべての列の最後の既知の値を返す
            return inputs[:, -1:, :].repeat(1, 24, 1)
        return inputs[:, -1:, self.label_index:self.label_index+1].repeat(1, 24, 1)

class RepeatBaseline(nn.Module):
    """
    マルチステップ予測用のベースラインモデル

    Args:
    label_index(int): 予測対象のインデックス
    """
    def __init__(self, label_index: int = None):
        super(RepeatBaseline, self).__init__()
        self.label_index = label_index

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        モデルの順伝播処理(最後の24時間分の既知データを次の24時間分の予測とする)

        Args:
            inputs(torch.Tensor): 入力データ

        Returns:
            torch.Tensor: 最後の24時間分のデータをそのまま返す出力
        """
        return inputs[:, -24:, self.label_index:self.label_index+1]
