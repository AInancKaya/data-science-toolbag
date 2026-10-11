import pandas as pd

def add_trading_indicators(df, price_col="close"):
    """Borsa/Finans verileri için teknik indikatörler üretir."""
    df = df.copy()
    
    # Hareketli Ortalamalar
    df["SMA_10"] = df[price_col].rolling(window=10).mean()
    df["SMA_50"] = df[price_col].rolling(window=50).mean()
    
    # Volatilite (Standart Sapma)
    df["Volatility_10"] = df[price_col].rolling(window=10).std()
    
    # Hedef Değişken (Geleceği Tahmin Etmek İçin) - Leakage Önlemli
    # N gün sonra fiyat artacak mı?
    df["Target_Next_Day_Return"] = df[price_col].shift(-1) / df[price_col] - 1
    df["Target_Up_Down"] = (df["Target_Next_Day_Return"] > 0).astype(int)
    
    return df.dropna()
