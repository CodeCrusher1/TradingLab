# Configurazione backtest

# Configurazione backtest

"""
Finestra di dialogo per la configurazione e l'esecuzione di backtest di strategie.
Permette di selezionare simboli, timeframe, intervallo di date e parametri della strategia.
"""
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QTabWidget, QSizePolicy, QScrollArea, QGroupBox, QCheckBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox, QProgressBar, QFileDialog,
    QWidget, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QIcon

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any

from ..charts import BacktestResultChart
from ..controls import (
    SymbolSelector, TimeframeSelector, DateRangeSelector, StrategySelector,
    PerformanceView
)
from ..styles import style_manager
from ...utils import app_logger
from ...config import SYMBOLS, TIMEFRAMES


class BacktestThread(QThread):
    """Thread per eseguire il backtest in background."""
    
    # Segnali
    progress = pyqtSignal(int)
    completed = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, symbol: str, timeframe: str, start_date: datetime, 
                 end_date: datetime, strategy: str, strategy_params: Dict[str, Any],
                 initial_capital: float, position_size: float, commission: float):
        """
        Inizializza il thread di backtest.
        
        Args:
            symbol: Simbolo da testare
            timeframe: Timeframe da utilizzare
            start_date: Data di inizio
            end_date: Data di fine
            strategy: Nome della strategia
            strategy_params: Parametri della strategia
            initial_capital: Capitale iniziale
            position_size: Dimensione posizione (%)
            commission: Commissione per operazione (%)
        """
        super().__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_date = start_date
        self.end_date = end_date
        self.strategy = strategy
        self.strategy_params = strategy_params
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission
    
    def run(self):
        """Esegue il backtest in background."""
        try:
            # Simulazione progresso
            for i in range(101):
                self.progress.emit(i)
                self.msleep(50)  # Simula il caricamento
            
            # Risultati del backtest di esempio
            # In un'implementazione reale, qui dovrebbe essere eseguito
            # il vero backtest utilizzando i parametri forniti
            
            # Crea una curva equity di esempio
            days = (self.end_date - self.start_date).days
            
            # Crea dati fittizi
            dates = [self.start_date + timedelta(days=i) for i in range(days)]
            initial_equity = self.initial_capital
            
            # Simula equity curve con un po' di rumore
            np.random.seed(42)  # Per riproducibilità
            
            # Trend base (crescente, decrescente o laterale)
            trend = np.linspace(0, 0.2, days)  # Trend positivo
            
            # Aggiungi rumore
            noise = np.random.normal(0, 0.01, days)
            
            # Combina trend e rumore
            returns = trend + noise
            
            # Calcola equity cumulativa
            equity_values = initial_equity * np.cumprod(1 + returns)
            
            # Crea la curva equity
            equity_curve = [(dates[i], equity_values[i]) for i in range(days)]
            
            # Calcola drawdown
            peaks = np.maximum.accumulate(equity_values)
            drawdown_values = (equity_values - peaks) / peaks * 100
            
            # Crea la curva drawdown
            drawdown_curve = [(dates[i], abs(drawdown_values[i])) for i in range(days)]
            
            # Calcola metriche di performance
            final_capital = equity_values[-1]
            total_return = (final_capital - initial_equity) / initial_equity * 100
            
            # Simula trade
            n_trades = int(days / 7)  # Circa un trade ogni 7 giorni
            win_rate = 0.65  # 65% di trade vincenti
            winning_trades = int(n_trades * win_rate)
            losing_trades = n_trades - winning_trades
            
            # Genera trade individuali
            trades = []
            for i in range(n_trades):
                is_win = i < winning_trades
                entry_date = self.start_date + timedelta(days=np.random.randint(0, days-1))
                exit_date = entry_date + timedelta(days=np.random.randint(1, 10))
                if exit_date > self.end_date:
                    exit_date = self.end_date
                
                direction = 1 if np.random.random() > 0.5 else -1
                entry_price = 100 + np.random.random() * 20
                
                if is_win:
                    exit_price = entry_price * (1 + 0.03 * direction)
                    profit = direction * (exit_price - entry_price) / entry_price * 100
                else:
                    exit_price = entry_price * (1 - 0.015 * direction)
                    profit = direction * (exit_price - entry_price) / entry_price * 100
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'direction': 'Long' if direction > 0 else 'Short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_loss': profit,
                    'profit_loss_amount': initial_equity * self.position_size / 100 * profit / 100
                })
            
            # Calcola metriche dettagliate
            profit_factor = 2.1  # Rapporto tra profitti e perdite
            sharpe_ratio = 1.5
            max_drawdown = abs(np.min(drawdown_values))
            
            # Prepara i risultati
            results = {
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'start_date': self.start_date.strftime('%d/%m/%Y'),
                'end_date': self.end_date.strftime('%d/%m/%Y'),
                'strategy_name': self.strategy,
                'strategy_params': self.strategy_params,
                'initial_capital': initial_equity,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_trades': n_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'equity_curve': equity_curve,
                'drawdown_curve': drawdown_curve,
                'trades': trades
            }
            
            # Emetti il segnale di completamento con i risultati
            self.completed.emit(results)
            
        except Exception as e:
            app_logger.error(f"Errore durante il backtest: {e}")
            self.error.emit(f"Errore durante il backtest: {e}")


class BacktestConfigTab(QWidget):
    """Tab per la configurazione dei parametri di backtest."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab di configurazione.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Usiamo un widget scrollabile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Widget contenitore
        container = QWidget()
        container_layout = QVBoxLayout(container)
        
        # Parametri del backtest
        # Gruppo simbolo e timeframe
        symbol_group = QGroupBox("Simbolo e Timeframe")
        symbol_layout = QHBoxLayout()
        
        self.symbol_selector = SymbolSelector()
        self.timeframe_selector = TimeframeSelector()
        
        symbol_layout.addWidget(self.symbol_selector)
        symbol_layout.addWidget(self.timeframe_selector)
        
        symbol_group.setLayout(symbol_layout)
        container_layout.addWidget(symbol_group)
        
        # Gruppo intervallo di date
        date_group = QGroupBox("Intervallo di Date")
        date_layout = QVBoxLayout()
        
        self.date_selector = DateRangeSelector()
        
        date_layout.addWidget(self.date_selector)
        
        date_group.setLayout(date_layout)
        container_layout.addWidget(date_group)
        
        # Gruppo strategia
        strategy_group = QGroupBox("Strategia di Trading")
        strategy_layout = QVBoxLayout()
        
        self.strategy_selector = StrategySelector()
        
        strategy_layout.addWidget(self.strategy_selector)
        
        strategy_group.setLayout(strategy_layout)
        container_layout.addWidget(strategy_group)
        
        # Gruppo parametri di trading
        trading_group = QGroupBox("Parametri di Trading")
        trading_layout = QGridLayout()
        
        # Capitale iniziale
        trading_layout.addWidget(QLabel("Capitale iniziale:"), 0, 0)
        self.capital_spin = QDoubleSpinBox()
        self.capital_spin.setRange(100, 1000000)
        self.capital_spin.setValue(10000)
        self.capital_spin.setPrefix("$ ")
        self.capital_spin.setSingleStep(1000)
        trading_layout.addWidget(self.capital_spin, 0, 1)
        
        # Dimensione posizione
        trading_layout.addWidget(QLabel("Dimensione posizione:"), 1, 0)
        self.position_size_spin = QDoubleSpinBox()
        self.position_size_spin.setRange(0.1, 100)
        self.position_size_spin.setValue(2)
        self.position_size_spin.setSuffix(" %")
        self.position_size_spin.setSingleStep(0.1)
        trading_layout.addWidget(self.position_size_spin, 1, 1)
        
        # Commissioni
        trading_layout.addWidget(QLabel("Commissioni:"), 2, 0)
        self.commission_spin = QDoubleSpinBox()
        self.commission_spin.setRange(0, 5)
        self.commission_spin.setValue(0.1)
        self.commission_spin.setSuffix(" %")
        self.commission_spin.setSingleStep(0.01)
        trading_layout.addWidget(self.commission_spin, 2, 1)
        
        # Opzioni aggiuntive
        trading_layout.addWidget(QLabel("Opzioni:"), 3, 0)
        options_layout = QVBoxLayout()
        
        self.slippage_check = QCheckBox("Includi slippage")
        self.slippage_check.setChecked(True)
        options_layout.addWidget(self.slippage_check)
        
        self.compounding_check = QCheckBox("Reinvestimento profitti")
        self.compounding_check.setChecked(True)
        options_layout.addWidget(self.compounding_check)
        
        trading_layout.addLayout(options_layout, 3, 1)
        
        trading_group.setLayout(trading_layout)
        container_layout.addWidget(trading_group)
        
        # Gruppo opzioni avanzate
        advanced_group = QGroupBox("Opzioni Avanzate")
        advanced_layout = QGridLayout()
        
        # Tipo di backtest
        advanced_layout.addWidget(QLabel("Tipo di backtest:"), 0, 0)
        self.backtest_type_combo = QComboBox()
        self.backtest_type_combo.addItem("Standard")
        self.backtest_type_combo.addItem("Monte Carlo")
        self.backtest_type_combo.addItem("Walk-Forward")
        advanced_layout.addWidget(self.backtest_type_combo, 0, 1)
        
        # Tipo di esecuzione
        advanced_layout.addWidget(QLabel("Esecuzione:"), 1, 0)
        self.execution_combo = QComboBox()
        self.execution_combo.addItem("Alla chiusura")
        self.execution_combo.addItem("Al prossimo tick")
        self.execution_combo.addItem("Realistic")
        advanced_layout.addWidget(self.execution_combo, 1, 1)
        
        advanced_group.setLayout(advanced_layout)
        container_layout.addWidget(advanced_group)
        
        # Aggiungi uno stretcher alla fine
        container_layout.addStretch()
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Ottiene la configurazione del backtest.
        
        Returns:
            Dizionario con i parametri di configurazione
        """
        config = {
            'symbol': self.symbol_selector.get_selected_symbol(),
            'timeframe': self.timeframe_selector.get_selected_timeframe(),
            'start_date': self.date_selector.get_date_range()[0],
            'end_date': self.date_selector.get_date_range()[1],
            'strategy': self.strategy_selector.get_selected_strategy(),
            'strategy_params': self.strategy_selector.get_strategy_params(),
            'initial_capital': self.capital_spin.value(),
            'position_size': self.position_size_spin.value(),
            'commission': self.commission_spin.value(),
            'slippage': self.slippage_check.isChecked(),
            'compounding': self.compounding_check.isChecked(),
            'backtest_type': self.backtest_type_combo.currentText().lower(),
            'execution_type': self.execution_combo.currentText().lower()
        }
        
        return config


class BacktestResultTab(QWidget):
    """Tab per visualizzare i risultati del backtest."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab dei risultati.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.results = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Grafico dei risultati
        self.chart = BacktestResultChart()
        
        # Performance
        self.performance_view = PerformanceView()
        
        # Aggiungi al layout
        layout.addWidget(self.chart, stretch=7)
        layout.addWidget(self.performance_view, stretch=3)
        
        self.setLayout(layout)
    
    def update_results(self, results: Dict[str, Any]):
        """
        Aggiorna i risultati visualizzati.
        
        Args:
            results: Dizionario con i risultati del backtest
        """
        self.results = results
        
        # Aggiorna il grafico
        if 'equity_curve' in results and 'drawdown_curve' in results:
            self.chart.set_backtest_results(results)
        
        # Aggiorna la vista performance
        self.performance_view.update_performance(results)


class BacktestTradesTab(QWidget):
    """Tab per visualizzare i singoli trade del backtest."""
    
    def __init__(self, parent=None):
        """
        Inizializza il tab dei trade.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.trades = []
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Tabella dei trade
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels([
            "Entrata", "Uscita", "Direzione", "Prezzo Entrata", 
            "Prezzo Uscita", "P/L %", "P/L $"
        ])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trades_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.trades_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        layout.addWidget(self.trades_table)
        
        # Statistiche dei trade
        stats_group = QGroupBox("Statistiche Trade")
        stats_layout = QGridLayout()
        
        stats_layout.addWidget(QLabel("Numero di trade:"), 0, 0)
        self.n_trades_label = QLabel("0")
        stats_layout.addWidget(self.n_trades_label, 0, 1)
        
        stats_layout.addWidget(QLabel("Trade vincenti:"), 1, 0)
        self.winning_trades_label = QLabel("0")
        stats_layout.addWidget(self.winning_trades_label, 1, 1)
        
        stats_layout.addWidget(QLabel("Trade perdenti:"), 2, 0)
        self.losing_trades_label = QLabel("0")
        stats_layout.addWidget(self.losing_trades_label, 2, 1)
        
        stats_layout.addWidget(QLabel("Profitto medio:"), 0, 2)
        self.avg_profit_label = QLabel("0%")
        stats_layout.addWidget(self.avg_profit_label, 0, 3)
        
        stats_layout.addWidget(QLabel("Perdita media:"), 1, 2)
        self.avg_loss_label = QLabel("0%")
        stats_layout.addWidget(self.avg_loss_label, 1, 3)
        
        stats_layout.addWidget(QLabel("Trade massimo:"), 2, 2)
        self.max_trade_label = QLabel("0%")
        stats_layout.addWidget(self.max_trade_label, 2, 3)
        
        stats_group.setLayout(stats_layout)
        
        layout.addWidget(stats_group)
        
        self.setLayout(layout)
    
    def update_trades(self, trades: List[Dict[str, Any]]):
        """
        Aggiorna i trade visualizzati.
        
        Args:
            trades: Lista di dizionari con i dettagli dei trade
        """
        self.trades = trades
        
        # Pulisci la tabella
        self.trades_table.setRowCount(0)
        
        if not trades:
            # Aggiorna le statistiche
            self.n_trades_label.setText("0")
            self.winning_trades_label.setText("0")
            self.losing_trades_label.setText("0")
            self.avg_profit_label.setText("0%")
            self.avg_loss_label.setText("0%")
            self.max_trade_label.setText("0%")
            return
        
        # Aggiungi righe alla tabella
        for row, trade in enumerate(trades):
            self.trades_table.insertRow(row)
            
            # Data entrata
            entry_date = trade["entry_date"]
            if isinstance(entry_date, str):
                entry_date_str = entry_date
            else:
                entry_date_str = entry_date.strftime("%d/%m/%Y")
            
            self.trades_table.setItem(row, 0, QTableWidgetItem(entry_date_str))
            
            # Data uscita
            exit_date = trade["exit_date"]
            if isinstance(exit_date, str):
                exit_date_str = exit_date
            else:
                exit_date_str = exit_date.strftime("%d/%m/%Y")
            
            self.trades_table.setItem(row, 1, QTableWidgetItem(exit_date_str))
            
            # Direzione
            direction = trade["direction"]
            direction_item = QTableWidgetItem(direction)
            
            if direction == "Long":
                direction_item.setForeground(style_manager.colors.SUCCESS)
            else:
                direction_item.setForeground(style_manager.colors.ERROR)
            
            self.trades_table.setItem(row, 2, direction_item)
            
            # Prezzo entrata
            entry_price = trade["entry_price"]
            self.trades_table.setItem(row, 3, QTableWidgetItem(f"{entry_price:.2f}"))
            
            # Prezzo uscita
            exit_price = trade["exit_price"]
            self.trades_table.setItem(row, 4, QTableWidgetItem(f"{exit_price:.2f}"))
            
            # P/L %
            profit_loss = trade["profit_loss"]
            profit_loss_item = QTableWidgetItem(f"{profit_loss:+.2f}%")
            
            if profit_loss > 0:
                profit_loss_item.setForeground(style_manager.colors.SUCCESS)
            elif profit_loss < 0:
                profit_loss_item.setForeground(style_manager.colors.ERROR)
            
            self.trades_table.setItem(row, 5, profit_loss_item)
            
            # P/L $
            profit_loss_amount = trade["profit_loss_amount"]
            profit_loss_amount_item = QTableWidgetItem(f"{profit_loss_amount:+.2f}")
            
            if profit_loss_amount > 0:
                profit_loss_amount_item.setForeground(style_manager.colors.SUCCESS)
            elif profit_loss_amount < 0:
                profit_loss_amount_item.setForeground(style_manager.colors.ERROR)
            
            self.trades_table.setItem(row, 6, profit_loss_amount_item)
        
        # Calcola statistiche
        winning_trades = [t for t in trades if t["profit_loss"] > 0]
        losing_trades = [t for t in trades if t["profit_loss"] <= 0]
        
        n_trades = len(trades)
        n_winning = len(winning_trades)
        n_losing = len(losing_trades)
        
        avg_profit = np.mean([t["profit_loss"] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t["profit_loss"] for t in losing_trades]) if losing_trades else 0
        max_profit = np.max([t["profit_loss"] for t in trades]) if trades else 0
        
        # Aggiorna le statistiche
        self.n_trades_label.setText(str(n_trades))
        self.winning_trades_label.setText(str(n_winning))
        self.losing_trades_label.setText(str(n_losing))
        self.avg_profit_label.setText(f"{avg_profit:+.2f}%")
        self.avg_loss_label.setText(f"{avg_loss:+.2f}%")
        self.max_trade_label.setText(f"{max_profit:+.2f}%")


class BacktestDialog(QDialog):
    """Finestra di dialogo per configurare ed eseguire un backtest."""
    
    def __init__(self, parent=None):
        """
        Inizializza la finestra di dialogo.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.backtest_thread = None
        self.results = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia della finestra di dialogo."""
        self.setWindowTitle("Backtest")
        self.setMinimumSize(800, 600)
        
        # Layout principale
        layout = QVBoxLayout(self)
        
        # Tab widget per le diverse sezioni
        self.tab_widget = QTabWidget()
        
        # Tab di configurazione
        self.config_tab = BacktestConfigTab()
        self.tab_widget.addTab(self.config_tab, "Configurazione")
        
        # Tab dei risultati
        self.result_tab = BacktestResultTab()
        self.tab_widget.addTab(self.result_tab, "Risultati")
        
        # Tab dei trade
        self.trades_tab = BacktestTradesTab()
        self.tab_widget.addTab(self.trades_tab, "Trade")
        
        # Aggiungi il tab widget al layout
        layout.addWidget(self.tab_widget)
        
        # Barra di progresso
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Pulsanti
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Esegui Backtest")
        self.run_button.clicked.connect(self.run_backtest)
        
        self.export_button = QPushButton("Esporta Risultati")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        
        self.close_button = QPushButton("Chiudi")
        self.close_button.clicked.connect(self.close)
        
        button_layout.addWidget(self.run_button)
        button_layout.addWidget(self.export_button)
        button_layout.addStretch()
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def run_backtest(self):
        """Esegue il backtest con i parametri configurati."""
        # Ottieni la configurazione
        config = self.config_tab.get_config()
        
        # Mostra la barra di progresso
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Disabilita il pulsante di esecuzione
        self.run_button.setEnabled(False)
        
        # Crea e avvia il thread di backtest
        self.backtest_thread = BacktestThread(
            symbol=config['symbol'],
            timeframe=config['timeframe'],
            start_date=config['start_date'],
            end_date=config['end_date'],
            strategy=config['strategy'],
            strategy_params=config['strategy_params'],
            initial_capital=config['initial_capital'],
            position_size=config['position_size'],
            commission=config['commission']
        )
        
        # Connetti i segnali
        self.backtest_thread.progress.connect(self.update_progress)
        self.backtest_thread.completed.connect(self.backtest_completed)
        self.backtest_thread.error.connect(self.backtest_error)
        
        # Avvia il thread
        self.backtest_thread.start()
    
    def update_progress(self, value: int):
        """
        Aggiorna la barra di progresso.
        
        Args:
            value: Percentuale di completamento
        """
        self.progress_bar.setValue(value)
    
    def backtest_completed(self, results: Dict[str, Any]):
        """
        Gestisce il completamento del backtest.
        
        Args:
            results: Dizionario con i risultati del backtest
        """
        self.results = results
        
        # Aggiorna i tab con i risultati
        self.result_tab.update_results(results)
        
        if 'trades' in results:
            self.trades_tab.update_trades(results['trades'])
        
        # Passa al tab dei risultati
        self.tab_widget.setCurrentIndex(1)
        
        # Nascondi la barra di progresso
        self.progress_bar.setVisible(False)
        
        # Riabilita il pulsante di esecuzione
        self.run_button.setEnabled(True)
        
        # Abilita il pulsante di esportazione
        self.export_button.setEnabled(True)
        
        # Mostra un messaggio
        QMessageBox.information(
            self,
            "Backtest Completato",
            f"Backtest completato con successo.\n\n"
            f"Rendimento totale: {results['total_return']:.2f}%\n"
            f"Win Rate: {results['win_rate']:.2f}%\n"
            f"Profit Factor: {results['profit_factor']:.2f}"
        )
    
    def backtest_error(self, error_msg: str):
        """
        Gestisce gli errori durante il backtest.
        
        Args:
            error_msg: Messaggio di errore
        """
        # Nascondi la barra di progresso
        self.progress_bar.setVisible(False)
        
        # Riabilita il pulsante di esecuzione
        self.run_button.setEnabled(True)
        
        # Mostra un messaggio di errore
        QMessageBox.critical(
            self,
            "Errore Backtest",
            f"Si è verificato un errore durante il backtest:\n{error_msg}"
        )
    
    def export_results(self):
        """Esporta i risultati del backtest."""
        if not self.results:
            return
        
        # Mostra una finestra di dialogo per selezionare il file
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Esporta Risultati",
            "",
            "CSV (*.csv);;Excel (*.xlsx);;HTML (*.html);;Tutti i file (*)"
        )
        
        if not filename:
            return
        
        try:
            # Esporta i risultati in base al tipo di file
            ext = filename.split('.')[-1].lower()
            
            if ext == 'csv':
                # Esporta in CSV
                # Prima creiamo un DataFrame con i trade
                if 'trades' in self.results:
                    trades_df = pd.DataFrame(self.results['trades'])
                    trades_df.to_csv(filename, index=False)
            
            elif ext == 'xlsx':
                # Esporta in Excel
                with pd.ExcelWriter(filename) as writer:
                    # Foglio dei trade
                    if 'trades' in self.results:
                        trades_df = pd.DataFrame(self.results['trades'])
                        trades_df.to_excel(writer, sheet_name='Trades', index=False)
                    
                    # Foglio dei parametri
                    params = {
                        'Simbolo': self.results['symbol'],
                        'Timeframe': self.results['timeframe'],
                        'Data Inizio': self.results['start_date'],
                        'Data Fine': self.results['end_date'],
                        'Strategia': self.results['strategy_name'],
                        'Capitale Iniziale': self.results['initial_capital'],
                        'Capitale Finale': self.results['final_capital'],
                        'Rendimento Totale (%)': self.results['total_return'],
                        'Win Rate (%)': self.results['win_rate'],
                        'Profit Factor': self.results['profit_factor'],
                        'Sharpe Ratio': self.results['sharpe_ratio'],
                        'Max Drawdown (%)': self.results['max_drawdown'],
                        'Numero Trade': self.results['total_trades'],
                    }
                    params_df = pd.DataFrame(list(params.items()), columns=['Parametro', 'Valore'])
                    params_df.to_excel(writer, sheet_name='Parametri', index=False)
            
            elif ext == 'html':
                # Esporta in HTML
                # Crea un report HTML semplice
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Backtest Report - {self.results['symbol']} {self.results['timeframe']}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; }}
                        .section {{ margin-bottom: 20px; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; }}
                        th {{ background-color: #f2f2f2; }}
                        .positive {{ color: green; }}
                        .negative {{ color: red; }}
                    </style>
                </head>
                <body>
                    <h1>Backtest Report</h1>
                    <div class="section">
                        <h2>Parametri</h2>
                        <table>
                            <tr><th>Parametro</th><th>Valore</th></tr>
                            <tr><td>Simbolo</td><td>{self.results['symbol']}</td></tr>
                            <tr><td>Timeframe</td><td>{self.results['timeframe']}</td></tr>
                            <tr><td>Data Inizio</td><td>{self.results['start_date']}</td></tr>
                            <tr><td>Data Fine</td><td>{self.results['end_date']}</td></tr>
                            <tr><td>Strategia</td><td>{self.results['strategy_name']}</td></tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Risultati</h2>
                        <table>
                            <tr><th>Metrica</th><th>Valore</th></tr>
                            <tr><td>Capitale Iniziale</td><td>${self.results['initial_capital']:.2f}</td></tr>
                            <tr><td>Capitale Finale</td><td>${self.results['final_capital']:.2f}</td></tr>
                            <tr><td>Rendimento Totale</td><td class="{'positive' if self.results['total_return'] >= 0 else 'negative'}">{self.results['total_return']:.2f}%</td></tr>
                            <tr><td>Win Rate</td><td>{self.results['win_rate']:.2f}%</td></tr>
                            <tr><td>Profit Factor</td><td>{self.results['profit_factor']:.2f}</td></tr>
                            <tr><td>Sharpe Ratio</td><td>{self.results['sharpe_ratio']:.2f}</td></tr>
                            <tr><td>Max Drawdown</td><td>{self.results['max_drawdown']:.2f}%</td></tr>
                            <tr><td>Numero Trade</td><td>{self.results['total_trades']}</td></tr>
                        </table>
                    </div>
                """
                
                # Aggiungi tabella dei trade se disponibile
                if 'trades' in self.results:
                    html_content += """
                    <div class="section">
                        <h2>Trade</h2>
                        <table>
                            <tr>
                                <th>Entrata</th>
                                <th>Uscita</th>
                                <th>Direzione</th>
                                <th>Prezzo Entrata</th>
                                <th>Prezzo Uscita</th>
                                <th>P/L %</th>
                                <th>P/L $</th>
                            </tr>
                    """
                    
                    for trade in self.results['trades']:
                        entry_date = trade["entry_date"]
                        exit_date = trade["exit_date"]
                        
                        if isinstance(entry_date, datetime):
                            entry_date = entry_date.strftime("%d/%m/%Y")
                        
                        if isinstance(exit_date, datetime):
                            exit_date = exit_date.strftime("%d/%m/%Y")
                        
                        profit_class = "positive" if trade["profit_loss"] > 0 else "negative"
                        
                        html_content += f"""
                            <tr>
                                <td>{entry_date}</td>
                                <td>{exit_date}</td>
                                <td>{trade["direction"]}</td>
                                <td>{trade["entry_price"]:.2f}</td>
                                <td>{trade["exit_price"]:.2f}</td>
                                <td class="{profit_class}">{trade["profit_loss"]:+.2f}%</td>
                                <td class="{profit_class}">{trade["profit_loss_amount"]:+.2f}</td>
                            </tr>
                        """
                    
                    html_content += """
                        </table>
                    </div>
                    """
                
                html_content += """
                </body>
                </html>
                """
                
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            else:
                # Formato non supportato, usa CSV come fallback
                if 'trades' in self.results:
                    trades_df = pd.DataFrame(self.results['trades'])
                    trades_df.to_csv(filename, index=False)
            
            QMessageBox.information(
                self,
                "Esportazione Completata",
                f"I risultati sono stati esportati in:\n{filename}"
            )
            
        except Exception as e:
            app_logger.error(f"Errore durante l'esportazione dei risultati: {e}")
            QMessageBox.critical(
                self,
                "Errore Esportazione",
                f"Si è verificato un errore durante l'esportazione dei risultati:\n{e}"
            )