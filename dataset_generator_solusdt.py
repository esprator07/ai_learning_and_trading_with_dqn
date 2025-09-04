import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import talib

class BinanceDataCollector:
    def __init__(self):
        # Futures API - API key gerekmez (public data)
        self.exchange = ccxt.binance({
            'sandbox': False,
            'rateLimit': 1200,  # Rate limit ayarı
            'options': {
                'defaultType': 'future',  # Futures API kullan
            }
        })
        
    def fetch_historical_data_batch(self, symbol='SOL/USDT:USDT', timeframe='1m', days=100):
        """
        Batch processing ile büyük veri setini çeker
        100 gün ≈ 144,000 mum → 144 request gerekir (1000'erli batch)
        """
        print(f"🔄 {symbol} için {days} günlük {timeframe} verileri batch olarak çekiliyor...")
        
        # Tarih hesaplama
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        since = self.exchange.parse8601(start_time.strftime('%Y-%m-%dT%H:%M:%S'))
        
        # Tahmini mum sayısı
        total_minutes = days * 24 * 60
        estimated_batches = (total_minutes // 1000) + 1
        print(f"📊 Tahmini {total_minutes:,} mum, {estimated_batches} batch ile çekilecek")
        
        all_ohlcv = []
        current_time = since
        batch_count = 0
        
        while current_time < self.exchange.milliseconds():
            try:
                batch_count += 1
                
                # 1000 mumluk batch çek
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe, 
                    current_time, 
                    limit=1000
                )
                
                if not ohlcv:
                    print("❌ Veri bulunamadı, durduruluyor")
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Son mum zamanından devam et
                current_time = ohlcv[-1][0] + 60000  # +1 dakika
                
                # İlerleme göster
                print(f"📦 Batch {batch_count}/{estimated_batches} | "
                      f"Bu batch: {len(ohlcv)} mum | "
                      f"Toplam: {len(all_ohlcv):,} mum")
                
                # Rate limit için bekle
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Batch {batch_count} hatası: {e}")
                print("⏳ 5 saniye bekleyip tekrar denenecek...")
                time.sleep(5)
                continue
        
        # DataFrame'e çevir
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        print(f"\n✅ Toplam {len(df):,} mum başarıyla çekildi!")
        return df
    
    def calculate_custom_returns(self, df):
        """
        Özel return hesaplamaları
        """
        print("🧮 Özel return hesaplamaları yapılıyor...")
        
        # Kopya oluştur - orijinal veriyi koru
        df = df.copy()
        
        # Önce return_close hesapla (çünkü return_open buna bağımlı)
        df['return_close'] = ((df['close'] - df['open']) / df['open'] * 100).round(4)
        
        # return_open: bir önceki mumun return_close değerine eşit
        df['return_open'] = df['return_close'].shift(1).round(4)
        
        # return_high: şu anki mum içinde open'dan high'a değişim (%)
        df['return_high'] = ((df['high'] - df['open']) / df['open'] * 100).round(4)
        
        # return_low: şu anki mum içinde open'dan low'a değişim (%) - düştüyse negatif
        df['return_low'] = ((df['low'] - df['open']) / df['open'] * 100).round(4)
        
        # return_volume: önceki mumla volume karşılaştırması (%)
        df['return_volume'] = ((df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100).round(4)
        
        # return2_volume: son 30 mumun volume ortalaması ile karşılaştırma (%)
        volume_ma30 = df['volume'].rolling(window=30).mean()
        df['return2_volume'] = ((df['volume'] - volume_ma30) / volume_ma30 * 100).round(4)
        
        return df
    
    def calculate_technical_indicators(self, df):
        """
        Teknik göstergeleri hesaplar - Fiyat-normalize edilmiş MACD
        """
        print("📈 Teknik göstergeler hesaplanıyor...")
        
        # RSI (14) - talib kullanarak
        df['rsi_14'] = talib.RSI(df['close'].values, timeperiod=14).round(4)
        
        # MACD hesaplamaları - Normalize edilmiş
        print("📊 Fiyat-normalize edilmiş MACD hesaplanıyor...")
        
        # Geleneksel MACD line, signal line, histogram
        macd_line, macd_signal, macd_histogram = talib.MACD(
            df['close'].values,
            fastperiod=12,    # Hızlı EMA
            slowperiod=26,    # Yavaş EMA  
            signalperiod=9    # Signal line EMA
        )
        
        # Fiyat ile normalize et (basis points - 0.01% units)
        # Bu sayede BTC/ETH/diğer coinler için aynı değer aralığı
        df['macd_line_norm'] = (macd_line / df['close'] * 10000).round(4)
        df['macd_signal_norm'] = (macd_signal / df['close'] * 10000).round(4)
        df['macd_hist_norm'] = (macd_histogram / df['close'] * 10000).round(4)
        
        # MACD momentum - normalize edilmiş (değişim hızı)
        df['macd_momentum_norm'] = df['macd_line_norm'].diff().round(4)
        
        # MACD divergence - line ile signal arasındaki normalize fark
        df['macd_divergence_norm'] = (df['macd_line_norm'] - df['macd_signal_norm']).round(4)
        
        # MACD cross signals (1=bullish, -1=bearish, 0=no cross)
        df['macd_cross'] = 0
        
        # Crossover hesapla - normalize değerlerle
        for i in range(1, len(df)):
            if pd.isna(df['macd_line_norm'].iloc[i-1]) or pd.isna(df['macd_line_norm'].iloc[i]):
                continue
                
            prev_diff = df['macd_line_norm'].iloc[i-1] - df['macd_signal_norm'].iloc[i-1]
            curr_diff = df['macd_line_norm'].iloc[i] - df['macd_signal_norm'].iloc[i]
            
            # Bullish cross: macd line signal line'ı yukarı kesiyor
            if prev_diff <= 0 and curr_diff > 0:
                df.loc[df.index[i], 'macd_cross'] = 1
            # Bearish cross: macd line signal line'ı aşağı kesiyor
            elif prev_diff >= 0 and curr_diff < 0:
                df.loc[df.index[i], 'macd_cross'] = -1
        
        # MACD histogram slope - normalize edilmiş (histogram trendin yönü)
        df['macd_hist_slope_norm'] = df['macd_hist_norm'].diff().round(4)
        
        # MACD strength - histogram'un mutlak değeri (momentum gücü)
        df['macd_strength_norm'] = np.abs(df['macd_hist_norm']).round(4)
        
        print("✅ Normalize edilmiş MACD indikatörleri eklendi:")
        print("   - macd_line_norm (basis points)")
        print("   - macd_signal_norm (basis points)")  
        print("   - macd_hist_norm (basis points)")
        print("   - macd_momentum_norm (değişim hızı)")
        print("   - macd_divergence_norm (line-signal fark)")
        print("   - macd_cross (crossover signals)")
        print("   - macd_hist_slope_norm (histogram eğim)")
        print("   - macd_strength_norm (momentum gücü)")
        
        print("\n🎯 Normalizasyon avantajları:")
        print("   ✅ BTC/ETH/tüm coinler için aynı model kullanılabilir")
        print("   ✅ Fiyat seviyesinden bağımsız MACD sinyalleri")
        print("   ✅ Basis points (0.01%) cinsinden ölçüm")
        
        return df
    
    def format_dataset(self, df):
        """
        Final dataset formatını hazırlar - MACD dahil
        """
        print("📋 Dataset formatlanıyor (MACD dahil)...")
        
        # Timestamp'i okunabilir tarihe çevir
        df['open_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = df['open_time'] + pd.Timedelta(minutes=1)
        
        # İstenen sütun sırası - Normalize edilmiş MACD
        final_columns = [
            'open_time', 'close_time', 'open', 'high', 'low', 'close', 'volume',
            'return_open', 'return_close', 'return_high', 'return_low', 
            'return_volume', 'return2_volume', 'rsi_14',
            # Normalize edilmiş MACD columns
            'macd_line_norm', 'macd_signal_norm', 'macd_hist_norm', 
            'macd_momentum_norm', 'macd_divergence_norm', 'macd_cross', 
            'macd_hist_slope_norm', 'macd_strength_norm'
        ]
        
        # Timestamp sütununu kaldır ve sıralama yap
        df = df.drop('timestamp', axis=1)
        df = df[final_columns]
        
        # NaN değerleri temizle (ilk ~35 mum RSI ve MACD hesaplanamaz)
        df = df.dropna()
        
        print(f"📊 NaN temizleme sonrası: {len(df):,} mum kaldı")
        
        return df
    
    def print_sample_data(self, df, sample_every=50):
        """
        Her N. mumda örnek veri gösterir - MACD dahil
        """
        print(f"\n📈 ÖRNEK VERİLER (Her {sample_every}. mum) - MACD DAHİL:")
        print("=" * 140)
        
        # Header - MACD dahil
        print(f"{'Tarih':<17} | {'Close':<10} | {'Ret_C':<8} | {'RSI':<6} | {'MACD':<8} | {'Signal':<8} | {'Hist':<8} | {'Cross':<6}")
        print("-" * 140)
        
        sample_data = df.iloc[::sample_every]
        
        for idx, row in sample_data.iterrows():
            date_str = row['open_time'].strftime('%Y-%m-%d %H:%M')
            close_price = f"${row['close']:.2f}"
            ret_close = f"{row['return_close']:.2f}%"
            rsi = f"{row['rsi_14']:.1f}" if not pd.isna(row['rsi_14']) else "N/A"
            
            # MACD değerleri
            macd = f"{row['macd_line']:.2f}" if not pd.isna(row['macd_line']) else "N/A"
            signal = f"{row['macd_signal']:.2f}" if not pd.isna(row['macd_signal']) else "N/A"
            hist = f"{row['macd_histogram']:.2f}" if not pd.isna(row['macd_histogram']) else "N/A"
            cross = "📈" if row['macd_cross'] == 1 else "📉" if row['macd_cross'] == -1 else "➡️"
            
            print(f"{date_str:<17} | {close_price:<10} | {ret_close:<8} | {rsi:<6} | {macd:<8} | {signal:<8} | {hist:<8} | {cross:<6}")

    def create_complete_dataset(self, symbol='SOL/USDT:USDT', days=100, save_path='solusdt_futures_100days.csv'):
        """
        Komplet veri seti oluşturma pipeline'ı - MACD dahil
        """
        print("🚀 SOL/USDT FUTURES VERİ SETİ OLUŞTURULUYOR (MACD DAHİL)...")
        print("=" * 70)
        
        # 1. Historical data çek (batch processing)
        df = self.fetch_historical_data_batch(symbol, '1m', days)
        
        # 2. Custom return hesaplamaları
        df = self.calculate_custom_returns(df)
        
        # 3. Teknik göstergeler (RSI + MACD)
        df = self.calculate_technical_indicators(df)
        
        # 4. Final format
        df = self.format_dataset(df)
        
        # 5. CSV'ye kaydet
        df.to_csv(save_path, index=False)
        
        # 6. Özet ve sample data
        print(f"\n✅ VERİ SETİ TAMAMLANDI!")
        print(f"📁 Dosya: {save_path}")
        print(f"📊 Toplam mum: {len(df):,}")
        print(f"📅 Tarih aralığı: {df['open_time'].min()} - {df['open_time'].max()}")
        print(f"💰 Fiyat aralığı: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"📈 Ortalama günlük volatilite: {df['return_close'].std():.2f}%")
        
        # MACD özet istatistikleri
        print(f"\n📊 MACD İSTATİSTİKLERİ:")
        print(f"   📈 MACD Line aralığı: {df['macd_line'].min():.2f} - {df['macd_line'].max():.2f}")
        print(f"   📊 Signal Line aralığı: {df['macd_signal'].min():.2f} - {df['macd_signal'].max():.2f}")
        print(f"   📉 Histogram aralığı: {df['macd_histogram'].min():.2f} - {df['macd_histogram'].max():.2f}")
        
        # Cross signal sayıları
        bullish_crosses = len(df[df['macd_cross'] == 1])
        bearish_crosses = len(df[df['macd_cross'] == -1])
        print(f"   📈 Bullish Cross: {bullish_crosses}")
        print(f"   📉 Bearish Cross: {bearish_crosses}")
        
        # Sample data göster
        self.print_sample_data(df)
        
        print(f"\n🎯 Dataset hazır! RL modeli için {df.shape[1]} feature mevcut.")
        print(f"🔥 Bir sonraki adım: RL Environment'a MACD feature'larını eklemek")
        
        return df
    
    def analyze_macd_signals(self, df):
        """
        MACD sinyallerinin performansını analiz et
        """
        print(f"\n🔍 MACD SİNYAL ANALİZİ:")
        print("=" * 50)
        
        # Cross sonrası N mumda fiyat değişimi
        for period in [5, 10, 30, 60]:  # 5, 10, 30, 60 dakika sonra
            bullish_performance = []
            bearish_performance = []
            
            for i in range(len(df) - period):
                if df['macd_cross'].iloc[i] == 1:  # Bullish cross
                    price_change = ((df['close'].iloc[i + period] - df['close'].iloc[i]) / df['close'].iloc[i]) * 100
                    bullish_performance.append(price_change)
                    
                elif df['macd_cross'].iloc[i] == -1:  # Bearish cross
                    price_change = ((df['close'].iloc[i + period] - df['close'].iloc[i]) / df['close'].iloc[i]) * 100
                    bearish_performance.append(price_change)
            
            if bullish_performance:
                avg_bullish = np.mean(bullish_performance)
                win_rate_bullish = (sum(1 for x in bullish_performance if x > 0) / len(bullish_performance)) * 100
            else:
                avg_bullish = win_rate_bullish = 0
                
            if bearish_performance:
                avg_bearish = np.mean(bearish_performance)
                win_rate_bearish = (sum(1 for x in bearish_performance if x < 0) / len(bearish_performance)) * 100
            else:
                avg_bearish = win_rate_bearish = 0
            
            print(f"📈 {period} dakika sonrası:")
            print(f"   🟢 Bullish Cross: Avg {avg_bullish:.2f}%, Win Rate: {win_rate_bullish:.1f}%")
            print(f"   🔴 Bearish Cross: Avg {avg_bearish:.2f}%, Win Rate: {win_rate_bearish:.1f}%")

# Çalıştır
if __name__ == "__main__":
    # Dataset oluşturucu
    collector = BinanceDataCollector()
    
    # 100 günlük SOLUSDT Futures dataset oluştur (MACD dahil)
    dataset = collector.create_complete_dataset(
        symbol='SOL/USDT:USDT',  # Futures sembolü
        days=100, 
        save_path='solusdt_futures_100days_with_macd.csv'
    )
    
    # MACD sinyal analizi
    collector.analyze_macd_signals(dataset)