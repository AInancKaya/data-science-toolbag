import polars as pl

def process_big_data(file_path):
    """Pandas'ın tıkandığı GB'larca büyüklükteki verilerde Polars ile ışık hızında işlem."""
    # Veriyi Tembel (Lazy) modda oku (RAM'i şişirmemek için)
    q = (
        pl.scan_csv(file_path)
        .filter(pl.col("age") > 18)
        .groupby("category")
        .agg(pl.col("salary").mean().alias("avg_salary"))
    )
    # İşlemi şimdi çalıştır
    df = q.collect()
    return df
