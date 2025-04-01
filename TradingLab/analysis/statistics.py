# Analisi statistiche

"""
Analisi statistica dei dati finanziari per il progetto TradingLab.
Questo modulo fornisce funzioni per analizzare statisticamente i prezzi e i pattern di mercato.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import math
from scipy import stats
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Importazioni dal modulo utils
from ..utils import app_logger, ProcessingError, time_it


class DescriptiveStats:
    """Statistiche descrittive per l'analisi dei dati di mercato."""
    
    @staticmethod
    def calculate_return_stats(df: pd.DataFrame, price_col: str = 'close') -> Dict[str, Any]:
        """
        Calcola le statistiche descrittive sui rendimenti.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            
        Returns:
            Dizionario con statistiche descrittive
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti logaritmici
        df_copy = df.copy()
        df_copy['log_returns'] = np.log(df_copy[price_col] / df_copy[price_col].shift(1))
        
        # Calcola le statistiche sui rendimenti
        stats_dict = {
            'mean': df_copy['log_returns'].mean(),
            'median': df_copy['log_returns'].median(),
            'std_dev': df_copy['log_returns'].std(),
            'annualized_volatility': df_copy['log_returns'].std() * np.sqrt(252),  # Assumendo dati giornalieri
            'skewness': df_copy['log_returns'].skew(),
            'kurtosis': df_copy['log_returns'].kurtosis(),
            'min': df_copy['log_returns'].min(),
            'max': df_copy['log_returns'].max(),
            'positive_days': sum(df_copy['log_returns'] > 0),
            'negative_days': sum(df_copy['log_returns'] < 0),
            'win_rate': sum(df_copy['log_returns'] > 0) / len(df_copy['log_returns'].dropna()),
            'avg_gain': df_copy['log_returns'][df_copy['log_returns'] > 0].mean(),
            'avg_loss': df_copy['log_returns'][df_copy['log_returns'] < 0].mean(),
            'gain_loss_ratio': abs(
                df_copy['log_returns'][df_copy['log_returns'] > 0].mean() / 
                df_copy['log_returns'][df_copy['log_returns'] < 0].mean()
            ) if df_copy['log_returns'][df_copy['log_returns'] < 0].mean() != 0 else np.nan,
            'sharpe_ratio': (
                df_copy['log_returns'].mean() / df_copy['log_returns'].std() * np.sqrt(252)
            ) if df_copy['log_returns'].std() != 0 else np.nan
        }
        
        # Calcola la normalità dei rendimenti (Test di Jarque-Bera)
        jb_stat, jb_pvalue = stats.jarque_bera(df_copy['log_returns'].dropna())
        stats_dict['jarque_bera_stat'] = jb_stat
        stats_dict['jarque_bera_pvalue'] = jb_pvalue
        stats_dict['is_normal'] = jb_pvalue > 0.05  # Valore p > 0.05 suggerisce normalità
        
        return stats_dict
    
    @staticmethod
    def calculate_drawdowns(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """
        Calcola i drawdown per l'analisi del rischio.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            
        Returns:
            DataFrame con analisi dei drawdown
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i massimi cumulativi
        df_copy = df.copy()
        df_copy['cumulative_max'] = df_copy[price_col].cummax()
        
        # Calcola i drawdown
        df_copy['drawdown'] = (df_copy[price_col] / df_copy['cumulative_max'] - 1) * 100
        
        # Identifica periodi di drawdown
        df_copy['is_drawdown'] = df_copy['drawdown'] < 0
        
        # Marca inizio di nuovi drawdown
        df_copy['new_drawdown'] = (
            (df_copy['is_drawdown'] != df_copy['is_drawdown'].shift(1)) & 
            df_copy['is_drawdown']
        ).astype(int)
        
        # Assegna un ID a ciascun drawdown
        df_copy['drawdown_id'] = df_copy['new_drawdown'].cumsum()
        
        # Rimuovi i non-drawdown
        drawdown_df = df_copy[df_copy['is_drawdown']].copy()
        
        # Calcola statistiche per ciascun drawdown
        drawdown_stats = []
        
        for dd_id, group in drawdown_df.groupby('drawdown_id'):
            if len(group) > 0:
                start_date = group.index[0]
                end_date = group.index[-1]
                max_drawdown = group['drawdown'].min()
                duration = len(group)
                
                drawdown_stats.append({
                    'drawdown_id': dd_id,
                    'start_date': start_date,
                    'end_date': end_date,
                    'max_drawdown': max_drawdown,
                    'duration': duration
                })
        
        # Crea DataFrame con le statistiche dei drawdown
        dd_stats_df = pd.DataFrame(drawdown_stats)
        
        # Ordina per drawdown massimo
        if not dd_stats_df.empty:
            dd_stats_df = dd_stats_df.sort_values('max_drawdown').reset_index(drop=True)
        
        return dd_stats_df
    
    @staticmethod
    def calculate_volatility_analysis(df: pd.DataFrame, price_col: str = 'close', 
                                    window: int = 21) -> pd.DataFrame:
        """
        Esegue un'analisi della volatilità.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            window: Finestra per calcolare la volatilità
            
        Returns:
            DataFrame con analisi della volatilità
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti logaritmici
        df_copy = df.copy()
        df_copy['log_returns'] = np.log(df_copy[price_col] / df_copy[price_col].shift(1))
        
        # Calcola volatilità a finestra mobile
        df_copy['volatility'] = df_copy['log_returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Calcola volatilità con EWMA (Exponentially Weighted Moving Average)
        df_copy['volatility_ewma'] = df_copy['log_returns'].ewm(span=window).std() * np.sqrt(252)
        
        # Calcola il "Volatility Ratio" (volatilità attuale / volatilità media)
        df_copy['volatility_ratio'] = df_copy['volatility'] / df_copy['volatility'].rolling(window=window*3).mean()
        
        # Classifica i regimi di volatilità
        vol_mean = df_copy['volatility'].mean()
        vol_std = df_copy['volatility'].std()
        
        df_copy['vol_regime'] = 'normale'
        df_copy.loc[df_copy['volatility'] > vol_mean + vol_std, 'vol_regime'] = 'alta'
        df_copy.loc[df_copy['volatility'] < vol_mean - vol_std, 'vol_regime'] = 'bassa'
        
        # Volatility Clustering (autocorrelazione dei rendimenti al quadrato)
        df_copy['returns_squared'] = df_copy['log_returns'] ** 2
        
        # Calcola lag-1 autocorrelation dei rendimenti al quadrato
        auto_corr = df_copy['returns_squared'].autocorr(lag=1)
        
        # Aggiunta del risultato come attributo
        df_copy.attrs['volatility_clustering'] = auto_corr
        df_copy.attrs['interpret_clustering'] = "Un'autocorrelazione positiva indica clustering di volatilità"
        
        return df_copy
    
    @staticmethod
    def calculate_correlation_matrix(prices_dict: Dict[str, pd.DataFrame], 
                                    price_col: str = 'close') -> pd.DataFrame:
        """
        Calcola la matrice di correlazione tra diversi asset.
        
        Args:
            prices_dict: Dizionario {nome_asset: dataframe}
            price_col: Nome della colonna del prezzo
            
        Returns:
            DataFrame con matrice di correlazione
        """
        # Verifica input
        if not prices_dict:
            raise ProcessingError("Il dizionario dei prezzi è vuoto")
        
        # Crea un DataFrame per i rendimenti
        returns_df = pd.DataFrame()
        
        # Aggiungi i rendimenti di ogni asset
        for asset_name, df in prices_dict.items():
            if price_col not in df.columns:
                raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame per {asset_name}")
            
            # Assicurati che l'indice sia temporale e ordinato
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                else:
                    raise ProcessingError(f"Nessun indice temporale trovato per {asset_name}")
            
            df = df.sort_index()
            
            # Calcola i rendimenti logaritmici
            returns = np.log(df[price_col] / df[price_col].shift(1))
            
            # Aggiungi al DataFrame dei rendimenti
            returns_df[asset_name] = returns
        
        # Calcola la matrice di correlazione
        corr_matrix = returns_df.corr()
        
        return corr_matrix


class TimeSeriesAnalysis:
    """Analisi delle serie temporali per dati finanziari."""
    
    @staticmethod
    def check_stationarity(df: pd.DataFrame, price_col: str = 'close') -> Dict[str, Any]:
        """
        Verifica la stazionarietà di una serie temporale usando il test ADF.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            
        Returns:
            Dizionario con i risultati del test
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Esegui il test Augmented Dickey-Fuller sulla serie originale
        result_original = adfuller(df[price_col].dropna())
        
        # Calcola i rendimenti e testa anche quelli
        df_copy = df.copy()
        df_copy['returns'] = df_copy[price_col].pct_change()
        result_returns = adfuller(df_copy['returns'].dropna())
        
        # Prepara i risultati
        results = {
            'price_series': {
                'adf_statistic': result_original[0],
                'p_value': result_original[1],
                'critical_values': result_original[4],
                'is_stationary': result_original[1] < 0.05
            },
            'returns_series': {
                'adf_statistic': result_returns[0],
                'p_value': result_returns[1],
                'critical_values': result_returns[4],
                'is_stationary': result_returns[1] < 0.05
            }
        }
        
        return results
    
    @staticmethod
    def decompose_time_series(df: pd.DataFrame, price_col: str = 'close', 
                             period: Optional[int] = None, model: str = 'additive') -> Dict[str, pd.Series]:
        """
        Decompone una serie temporale nelle sue componenti.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            period: Periodo per la decomposizione stagionale (es. 252 per dati giornalieri annuali)
            model: Tipo di modello ('additive' o 'multiplicative')
            
        Returns:
            Dizionario con le componenti decomposte
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Assicurati che la serie sia una serie temporale
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ProcessingError("Nessun indice temporale trovato")
        
        # Stima il periodo se non fornito
        if period is None:
            # Se dati giornalieri, usa 252 (circa giorni di trading in un anno)
            if df.index.to_series().diff().median().days == 1:
                period = 252
            # Se dati settimanali, usa 52
            elif 6 <= df.index.to_series().diff().median().days <= 8:
                period = 52
            # Se dati mensili, usa 12
            elif 28 <= df.index.to_series().diff().median().days <= 31:
                period = 12
            # Altrimenti usa un periodo arbitrario
            else:
                period = min(len(df) // 2, 12)
        
        # Decomponi la serie
        decomposition = seasonal_decompose(df[price_col], model=model, period=period)
        
        # Estrai le componenti
        components = {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
        
        return components
    
    @staticmethod
    def fit_arima_model(df: pd.DataFrame, price_col: str = 'close', 
                       order: Tuple[int, int, int] = (1, 1, 1)) -> Dict[str, Any]:
        """
        Adatta un modello ARIMA alla serie temporale.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            order: Ordine del modello ARIMA (p, d, q)
            
        Returns:
            Dizionario con risultati del modello e previsioni
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Adatta il modello ARIMA
        model = ARIMA(df[price_col], order=order)
        model_fit = model.fit()
        
        # Genera previsioni in-sample
        predictions = model_fit.predict()
        
        # Calcola statistiche di errore
        mse = np.mean((predictions - df[price_col].iloc[len(df[price_col]) - len(predictions):]) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - df[price_col].iloc[len(df[price_col]) - len(predictions):]))
        
        # Previsioni future (5 periodi)
        forecast = model_fit.forecast(steps=5)
        
        # Prepara i risultati
        results = {
            'model_summary': model_fit.summary(),
            'predictions': predictions,
            'forecast': forecast,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'aic': model_fit.aic,
            'bic': model_fit.bic
        }
        
        return results
    
    @staticmethod
    def calculate_autocorrelation(df: pd.DataFrame, col: str = 'close', 
                                lags: int = 20) -> Dict[str, np.ndarray]:
        """
        Calcola l'autocorrelazione e l'autocorrelazione parziale.
        
        Args:
            df: DataFrame con dati
            col: Nome della colonna da analizzare
            lags: Numero di lag da calcolare
            
        Returns:
            Dizionario con valori di autocorrelazione
        """
        if col not in df.columns:
            raise ProcessingError(f"Colonna {col} non trovata nel DataFrame")
        
        # Calcola i rendimenti
        returns = df[col].pct_change().dropna()
        
        # Calcola ACF (AutoCorrelation Function)
        acf_values = acf(returns, nlags=lags, fft=True)
        
        # Calcola PACF (Partial AutoCorrelation Function)
        pacf_values = pacf(returns, nlags=lags)
        
        # Prepara i risultati
        results = {
            'acf': acf_values,
            'pacf': pacf_values,
            'lags': np.arange(lags + 1)
        }
        
        return results


class MarketEfficiencyAnalysis:
    """Analisi dell'efficienza di mercato."""
    
    @staticmethod
    def run_random_walk_test(df: pd.DataFrame, price_col: str = 'close') -> Dict[str, Any]:
        """
        Esegue il test della random walk (test di Ljung-Box).
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            
        Returns:
            Dizionario con risultati del test
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti logaritmici
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        # Test di Ljung-Box per autocorrelazione
        lb_stat, lb_pvalue = stats.acorr_ljungbox(log_returns, lags=[10], return_df=False)
        
        # Test runs per randomness
        # Calcola i segni dei rendimenti
        signs = np.sign(log_returns)
        
        # Conta le runs
        n_positive = sum(signs > 0)
        n_negative = sum(signs < 0)
        
        # Calcola il numero di run
        runs = 1
        for i in range(1, len(signs)):
            if signs[i] != signs[i-1]:
                runs += 1
        
        # Calcola il valore atteso e la varianza del numero di run
        expected_runs = 2 * n_positive * n_negative / (n_positive + n_negative) + 1
        var_runs = (2 * n_positive * n_negative * (2 * n_positive * n_negative - n_positive - n_negative)) / (
            (n_positive + n_negative)**2 * (n_positive + n_negative - 1)
        )
        
        # Calcola la statistica Z del test run
        z_stat = (runs - expected_runs) / np.sqrt(var_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # test a due code
        
        # Test della varianza della ratio
        k_values = [2, 4, 8, 16]
        vr_results = {}
        
        for k in k_values:
            # Calcola i rendimenti per intervalli di k
            returns_k = np.log(df[price_col].iloc[::k] / df[price_col].iloc[::k].shift(1)).dropna()
            
            # Calcola la varianza per i due intervalli
            var_1 = np.var(log_returns)
            var_k = np.var(returns_k)
            
            # Calcola la Variance Ratio
            vr = var_k / (k * var_1)
            
            vr_results[k] = vr
        
        # Prepara i risultati
        results = {
            'ljung_box': {
                'statistic': lb_stat[0],
                'p_value': lb_pvalue[0],
                'is_random_walk': lb_pvalue[0] > 0.05
            },
            'runs_test': {
                'runs': runs,
                'expected_runs': expected_runs,
                'z_statistic': z_stat,
                'p_value': p_value,
                'is_random': p_value > 0.05
            },
            'variance_ratio': vr_results,
            'is_efficient': lb_pvalue[0] > 0.05 and p_value > 0.05
        }
        
        return results
    
    @staticmethod
    def calculate_hurst_exponent(df: pd.DataFrame, price_col: str = 'close', 
                               max_lag: int = 20) -> Dict[str, Any]:
        """
        Calcola l'esponente di Hurst per testare la memoria a lungo termine.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            max_lag: Massimo lag per il calcolo
            
        Returns:
            Dizionario con risultati dell'analisi
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti logaritmici
        log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
        
        # Calcola la range rescaled statistic per diversi lag
        lags = range(2, max_lag)
        tau = []; rs = []
        
        for lag in lags:
            # Suddividi la serie in blocchi di dimensione lag
            n_blocks = int(len(log_returns) / lag)
            
            if n_blocks < 1:
                continue
            
            rs_values = []
            
            for i in range(n_blocks):
                # Estrai il blocco
                block = log_returns.iloc[i*lag:(i+1)*lag].values
                
                # Calcola la media e la deviazione standard
                mean = np.mean(block)
                std = np.std(block)
                
                if std == 0:
                    continue
                
                # Calcola la serie cumulativa aggiustata per la media
                adjusted_block = block - mean
                cumulative = np.cumsum(adjusted_block)
                
                # Calcola il range
                range_value = max(cumulative) - min(cumulative)
                
                # Calcola la R/S statistic
                rs_value = range_value / std
                
                rs_values.append(rs_value)
            
            if rs_values:
                rs.append(np.mean(rs_values))
                tau.append(lag)
        
        # Calcola l'esponente di Hurst usando regressione log-log
        if len(tau) > 1:
            x = np.log10(tau)
            y = np.log10(rs)
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            hurst_exponent = slope
        else:
            hurst_exponent = np.nan
            r_value = np.nan
            p_value = np.nan
            std_err = np.nan
        
        # Interpretazione dell'esponente di Hurst
        if np.isnan(hurst_exponent):
            interpretation = "Non è stato possibile calcolare l'esponente di Hurst"
        elif hurst_exponent < 0.5:
            interpretation = "Anti-persistente (mean-reverting)"
        elif hurst_exponent > 0.5:
            interpretation = "Persistente (trend-following)"
        else:
            interpretation = "Random walk / Brownian motion"
        
        # Prepara i risultati
        results = {
            'hurst_exponent': hurst_exponent,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_error': std_err,
            'interpretation': interpretation,
            'tau': tau,
            'rs': rs
        }
        
        return results
    
    @staticmethod
    def test_market_regimes(df: pd.DataFrame, price_col: str = 'close', 
                          window: int = 252) -> pd.DataFrame:
        """
        Identifica i regimi di mercato basati su analisi statistiche.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            window: Finestra per l'analisi a finestra mobile
            
        Returns:
            DataFrame con indicatori di regime di mercato
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti logaritmici
        df_copy = df.copy()
        df_copy['log_returns'] = np.log(df_copy[price_col] / df_copy[price_col].shift(1))
        
        # Calcola media mobile dei rendimenti
        df_copy['returns_ma'] = df_copy['log_returns'].rolling(window=window).mean()
        
        # Calcola volatilità mobile
        df_copy['volatility'] = df_copy['log_returns'].rolling(window=window).std()
        
        # Definisci i regimi di mercato
        # Regime 1: Trend rialzista con bassa volatilità (Bull Market)
        # Regime 2: Trend rialzista con alta volatilità (Bull Market volatile)
        # Regime 3: Trend ribassista con bassa volatilità (Bear Market)
        # Regime 4: Trend ribassista con alta volatilità (Bear Market volatile)
        
        # Calcola la media e la deviazione standard storiche dei rendimenti e della volatilità
        mean_return = df_copy['log_returns'].mean()
        mean_vol = df_copy['volatility'].mean()
        
        # Definisci i regimi
        df_copy['market_regime'] = 0
        
        # Regime 1: Bull Market stabile
        df_copy.loc[(df_copy['returns_ma'] > 0) & 
                  (df_copy['volatility'] <= mean_vol), 'market_regime'] = 1
        
        # Regime 2: Bull Market volatile
        df_copy.loc[(df_copy['returns_ma'] > 0) & 
                  (df_copy['volatility'] > mean_vol), 'market_regime'] = 2
        
        # Regime 3: Bear Market stabile
        df_copy.loc[(df_copy['returns_ma'] <= 0) & 
                  (df_copy['volatility'] <= mean_vol), 'market_regime'] = 3
        
        # Regime 4: Bear Market volatile
        df_copy.loc[(df_copy['returns_ma'] <= 0) & 
                  (df_copy['volatility'] > mean_vol), 'market_regime'] = 4
        
        # Traduzione numerica in categorie
        df_copy['regime_name'] = df_copy['market_regime'].map({
            0: 'Indefinito',
            1: 'Bull Market Stabile',
            2: 'Bull Market Volatile',
            3: 'Bear Market Stabile',
            4: 'Bear Market Volatile'
        })
        
        return df_copy


class EventStudyAnalysis:
    """Analisi degli eventi di mercato."""
    
    @staticmethod
    def analyze_event_impact(df: pd.DataFrame, event_dates: List[datetime], 
                           price_col: str = 'close', window: int = 10) -> Dict[str, Any]:
        """
        Analizza l'impatto di eventi specifici sui prezzi.
        
        Args:
            df: DataFrame con dati di prezzo
            event_dates: Lista di date degli eventi
            price_col: Nome della colonna del prezzo
            window: Finestra prima/dopo l'evento (giorni)
            
        Returns:
            Dizionario con analisi dell'impatto
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Assicurati che il DataFrame abbia un indice datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ProcessingError("Nessun indice temporale trovato")
        
        # Calcola i rendimenti
        df_copy = df.copy()
        df_copy['returns'] = df_copy[price_col].pct_change()
        
        # Inizializza risultati
        event_results = []
        
        for event_date in event_dates:
            try:
                # Trova l'indice più vicino alla data dell'evento
                if event_date not in df_copy.index:
                    nearest_idx = df_copy.index[df_copy.index.get_indexer([event_date], method='nearest')[0]]
                else:
                    nearest_idx = event_date
                
                event_idx = df_copy.index.get_loc(nearest_idx)
                
                # Estrai finestra prima e dopo l'evento
                start_idx = max(0, event_idx - window)
                end_idx = min(len(df_copy) - 1, event_idx + window)
                
                pre_event = df_copy.iloc[start_idx:event_idx]
                post_event = df_copy.iloc[event_idx+1:end_idx+1]
                
                # Calcola i rendimenti cumulativi
                pre_cumulative_return = (1 + pre_event['returns']).prod() - 1
                post_cumulative_return = (1 + post_event['returns']).prod() - 1
                
                # Calcola volatilità prima e dopo
                pre_volatility = pre_event['returns'].std()
                post_volatility = post_event['returns'].std()
                
                # Calcola il volume degli scambi (se disponibile)
                if 'volume' in df_copy.columns:
                    pre_volume = pre_event['volume'].mean()
                    post_volume = post_event['volume'].mean()
                    volume_change = (post_volume / pre_volume - 1) if pre_volume > 0 else np.nan
                else:
                    pre_volume = np.nan
                    post_volume = np.nan
                    volume_change = np.nan
                
                # Aggiungi risultati
                event_results.append({
                    'event_date': nearest_idx,
                    'pre_event_return': pre_cumulative_return,
                    'post_event_return': post_cumulative_return,
                    'net_impact': post_cumulative_return - pre_cumulative_return,
                    'pre_volatility': pre_volatility,
                    'post_volatility': post_volatility,
                    'volatility_change': post_volatility / pre_volatility - 1,
                    'pre_volume': pre_volume,
                    'post_volume': post_volume,
                    'volume_change': volume_change
                })
            except Exception as e:
                app_logger.warning(f"Errore nell'analisi dell'evento {event_date}: {str(e)}")
        
        # Crea DataFrame con i risultati
        results_df = pd.DataFrame(event_results)
        
        # Calcola statistiche aggregate
        avg_net_impact = results_df['net_impact'].mean()
        avg_vol_change = results_df['volatility_change'].mean()
        
        # Test statistico per significatività dell'impatto
        if len(results_df) > 1:
            t_stat, p_value = stats.ttest_1samp(results_df['net_impact'], 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # Prepara i risultati
        results = {
            'event_analysis': results_df,
            'aggregate': {
                'avg_net_impact': avg_net_impact,
                'avg_volatility_change': avg_vol_change,
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
            }
        }
        
        return results
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, price_col: str = 'close', 
                        window: int = 20, sigma: float = 3.0) -> pd.DataFrame:
        """
        Rileva anomalie nei prezzi usando Z-score o altre tecniche.
        
        Args:
            df: DataFrame con dati di prezzo
            price_col: Nome della colonna del prezzo
            window: Finestra per il calcolo
            sigma: Numero di deviazioni standard per considerare un'anomalia
            
        Returns:
            DataFrame con anomalie rilevate
        """
        if price_col not in df.columns:
            raise ProcessingError(f"Colonna {price_col} non trovata nel DataFrame")
        
        # Calcola i rendimenti
        df_copy = df.copy()
        df_copy['returns'] = df_copy[price_col].pct_change()
        
        # Calcola media mobile e deviazione standard
        df_copy['returns_mean'] = df_copy['returns'].rolling(window=window).mean()
        df_copy['returns_std'] = df_copy['returns'].rolling(window=window).std()
        
        # Calcola Z-score
        df_copy['z_score'] = (df_copy['returns'] - df_copy['returns_mean']) / df_copy['returns_std']
        
        # Identifica anomalie
        df_copy['is_anomaly'] = abs(df_copy['z_score']) > sigma
        
        # Classifica anomalie
        df_copy['anomaly_direction'] = 0
        df_copy.loc[df_copy['z_score'] > sigma, 'anomaly_direction'] = 1  # Anomalia positiva
        df_copy.loc[df_copy['z_score'] < -sigma, 'anomaly_direction'] = -1  # Anomalia negativa
        
        return df_copy


# Factory function per accedere alle classi di analisi statistica
def get_statistical_analyzers():
    """
    Ottiene un'istanza di ogni classe di analisi statistica.
    
    Returns:
        Dizionario con istanze di classi di analisi statistica
    """
    return {
        'descriptive': DescriptiveStats(),
        'time_series': TimeSeriesAnalysis(),
        'efficiency': MarketEfficiencyAnalysis(),
        'event_study': EventStudyAnalysis()
    }