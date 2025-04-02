# Dashboard principale

"""
Dashboard principale dell'applicazione TradingLab.
Visualizza un riepilogo delle informazioni chiave e widget interattivi.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGridLayout,
    QFrame, QScrollArea, QSizePolicy, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView
)
from PyQt6.QtCore import Qt, QSize, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor  # Aggiungo QColor per convertire le stringhe di colore

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple, Any

from .charts import ChartWidget, BacktestResultChart
from .controls import (
    MarketInfoView, TradeSignalView, PredictionResultView, PerformanceView
)
from .styles import style_manager
from ..utils import app_logger


class MarketSummaryWidget(QWidget):
    """Widget per visualizzare un riepilogo del mercato."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget di riepilogo del mercato.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.market_data = []  # Dati di mercato
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title = QLabel("Riepilogo Mercato")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        
        # Tabella mercato
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(4)
        self.market_table.setHorizontalHeaderLabels(["Simbolo", "Ultimo", "Var %", "Volume"])
        self.market_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.market_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.market_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        layout.addWidget(self.market_table)
        
        # Aggiorna i dati iniziali
        self.update_data([])
    
    def update_data(self, market_data: List[Dict[str, Any]]):
        """
        Aggiorna i dati di mercato.
        
        Args:
            market_data: Lista di dizionari con dati di mercato
        """
        self.market_data = market_data
        
        # Pulisci la tabella
        self.market_table.setRowCount(0)
        
        if not market_data:
            # Aggiungi alcune righe di esempio
            example_data = [
                {"symbol": "EURUSD", "price": 1.0825, "change": 0.15, "volume": 128500},
                {"symbol": "USDJPY", "price": 151.75, "change": -0.28, "volume": 98760},
                {"symbol": "BTCUSD", "price": 55420.50, "change": 2.34, "volume": 12345},
                {"symbol": "AAPL", "price": 178.25, "change": 0.45, "volume": 5642300},
                {"symbol": "MSFT", "price": 415.30, "change": -0.12, "volume": 3215600}
            ]
            market_data = example_data
        
        # Aggiungi righe alla tabella
        for row, data in enumerate(market_data):
            self.market_table.insertRow(row)
            
            # Simbolo
            symbol_item = QTableWidgetItem(data["symbol"])
            self.market_table.setItem(row, 0, symbol_item)
            
            # Prezzo
            price_item = QTableWidgetItem(f"{data['price']:.2f}")
            self.market_table.setItem(row, 1, price_item)
            
            # Variazione percentuale
            change = data["change"]
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            
            # Imposta colore in base alla variazione
            if change > 0:
                change_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif change < 0:
                change_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.market_table.setItem(row, 2, change_item)
            
            # Volume
            volume = data["volume"]
            
            # Formatta il volume
            if volume >= 1_000_000:
                volume_text = f"{volume/1_000_000:.2f}M"
            elif volume >= 1_000:
                volume_text = f"{volume/1_000:.2f}K"
            else:
                volume_text = str(volume)
            
            volume_item = QTableWidgetItem(volume_text)
            self.market_table.setItem(row, 3, volume_item)


class WatchlistWidget(QWidget):
    """Widget per visualizzare una watchlist personalizzata."""
    
    # Segnale emesso quando si seleziona un simbolo nella watchlist
    symbolSelected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il widget watchlist.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.watchlist = []  # Lista di simboli osservati
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title_layout = QHBoxLayout()
        
        title = QLabel("Watchlist")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        title_layout.addWidget(title)
        
        # Pulsante per aggiungere simboli
        add_button = QPushButton("+")
        add_button.setMaximumWidth(30)
        add_button.clicked.connect(self.add_symbol)
        title_layout.addWidget(add_button)
        
        layout.addLayout(title_layout)
        
        # Tabella watchlist
        self.watchlist_table = QTableWidget()
        self.watchlist_table.setColumnCount(5)
        self.watchlist_table.setHorizontalHeaderLabels(["Simbolo", "Ultimo", "Var %", "Segnale", "Rimuovi"])
        self.watchlist_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.watchlist_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.watchlist_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        # Connetti segnale di selezione
        self.watchlist_table.itemClicked.connect(self.on_item_clicked)
        
        layout.addWidget(self.watchlist_table)
        
        # Aggiorna i dati iniziali
        self.update_watchlist([])
    
    def update_watchlist(self, watchlist_data: List[Dict[str, Any]]):
        """
        Aggiorna i dati della watchlist.
        
        Args:
            watchlist_data: Lista di dizionari con dati della watchlist
        """
        self.watchlist = watchlist_data
        
        # Pulisci la tabella
        self.watchlist_table.setRowCount(0)
        
        if not watchlist_data:
            # Aggiungi alcune righe di esempio
            example_data = [
                {"symbol": "EURUSD", "price": 1.0825, "change": 0.15, "signal": "buy"},
                {"symbol": "BTCUSD", "price": 55420.50, "change": 2.34, "signal": "wait"},
                {"symbol": "AAPL", "price": 178.25, "change": 0.45, "signal": "sell"}
            ]
            watchlist_data = example_data
        
        # Aggiungi righe alla tabella
        for row, data in enumerate(watchlist_data):
            self.watchlist_table.insertRow(row)
            
            # Simbolo
            symbol_item = QTableWidgetItem(data["symbol"])
            self.watchlist_table.setItem(row, 0, symbol_item)
            
            # Prezzo
            price_item = QTableWidgetItem(f"{data['price']:.2f}")
            self.watchlist_table.setItem(row, 1, price_item)
            
            # Variazione percentuale
            change = data["change"]
            change_item = QTableWidgetItem(f"{change:+.2f}%")
            
            # Imposta colore in base alla variazione
            if change > 0:
                change_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif change < 0:
                change_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.watchlist_table.setItem(row, 2, change_item)
            
            # Segnale
            signal = data.get("signal", "wait")
            signal_text = {
                "buy": "ACQUISTA",
                "sell": "VENDI",
                "wait": "ATTENDI"
            }.get(signal, "")
            
            signal_item = QTableWidgetItem(signal_text)
            
            # Imposta colore in base al segnale
            if signal == "buy":
                signal_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif signal == "sell":
                signal_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.watchlist_table.setItem(row, 3, signal_item)
            
            # Pulsante rimuovi
            remove_item = QTableWidgetItem("❌")
            remove_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.watchlist_table.setItem(row, 4, remove_item)
    
    def add_symbol(self):
        """Aggiunge un nuovo simbolo alla watchlist."""
        # In un'implementazione reale, mostrerebbe una finestra di dialogo
        # per selezionare un simbolo da aggiungere
        
        # Per ora, aggiungiamo un simbolo di esempio
        new_symbol = {
            "symbol": "GOOGL",
            "price": 2145.30,
            "change": 1.28,
            "signal": "buy"
        }
        
        self.watchlist.append(new_symbol)
        self.update_watchlist(self.watchlist)
    
    def on_item_clicked(self, item):
        """
        Gestisce il click su un elemento della tabella.
        
        Args:
            item: Elemento cliccato
        """
        row = item.row()
        col = item.column()
        
        if col == 4:  # Colonna "Rimuovi"
            # Rimuovi il simbolo dalla watchlist
            self.watchlist.pop(row)
            self.update_watchlist(self.watchlist)
        else:
            # Ottieni il simbolo selezionato
            symbol = self.watchlist_table.item(row, 0).text()
            self.symbolSelected.emit(symbol)


class MiniChartWidget(QWidget):
    """Widget per visualizzare grafici in miniatura."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget di grafico in miniatura.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.chart_data = []  # Dati per i grafici
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title = QLabel("Grafici")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        
        # Layout a griglia per i grafici
        charts_layout = QGridLayout()
        
        # Crea 4 grafici di esempio
        for i in range(4):
            chart_frame = QFrame()
            chart_frame.setFrameShape(QFrame.Shape.StyledPanel)
            chart_frame.setMinimumHeight(150)
            
            chart_layout = QVBoxLayout(chart_frame)
            
            # Simbolo e timeframe
            symbol_label = QLabel(f"Simbolo {i+1}")
            font = symbol_label.font()
            font.setBold(True)
            symbol_label.setFont(font)
            chart_layout.addWidget(symbol_label)
            
            # Placeholder per il grafico
            chart_layout.addWidget(QLabel("Grafico qui"))
            
            # Posiziona nella griglia
            row = i // 2
            col = i % 2
            charts_layout.addWidget(chart_frame, row, col)
        
        layout.addLayout(charts_layout)


class CalendarWidget(QWidget):
    """Widget per visualizzare un calendario economico."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget calendario.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.events = []  # Eventi economici
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title = QLabel("Calendario Economico")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        
        # Tabella eventi
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(4)
        self.events_table.setHorizontalHeaderLabels(["Ora", "Valuta", "Evento", "Impatto"])
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.events_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self.events_table)
        
        # Aggiorna i dati iniziali
        self.update_events([])
    
    def update_events(self, events: List[Dict[str, Any]]):
        """
        Aggiorna gli eventi economici.
        
        Args:
            events: Lista di dizionari con eventi economici
        """
        self.events = events
        
        # Pulisci la tabella
        self.events_table.setRowCount(0)
        
        if not events:
            # Aggiungi alcune righe di esempio
            now = datetime.now()
            example_events = [
                {
                    "time": now.replace(hour=8, minute=30),
                    "currency": "EUR",
                    "event": "PIL Germania",
                    "impact": "high"
                },
                {
                    "time": now.replace(hour=14, minute=15),
                    "currency": "USD",
                    "event": "Produzione Industriale",
                    "impact": "medium"
                },
                {
                    "time": now.replace(hour=16, minute=0),
                    "currency": "USD",
                    "event": "Discorso FED",
                    "impact": "high"
                }
            ]
            events = example_events
        
        # Ordina eventi per orario
        events = sorted(events, key=lambda x: x["time"])
        
        # Aggiungi righe alla tabella
        for row, event in enumerate(events):
            self.events_table.insertRow(row)
            
            # Ora
            time_str = event["time"].strftime("%H:%M")
            time_item = QTableWidgetItem(time_str)
            self.events_table.setItem(row, 0, time_item)
            
            # Valuta
            currency_item = QTableWidgetItem(event["currency"])
            self.events_table.setItem(row, 1, currency_item)
            
            # Evento
            event_item = QTableWidgetItem(event["event"])
            self.events_table.setItem(row, 2, event_item)
            
            # Impatto
            impact = event["impact"]
            impact_item = QTableWidgetItem(impact.upper())
            
            # Imposta colore in base all'impatto
            if impact == "high":
                impact_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            elif impact == "medium":
                impact_item.setForeground(QColor(style_manager.colors.WARNING))  # Converto la stringa in QColor
            else:
                impact_item.setForeground(QColor(style_manager.colors.INFO))  # Converto la stringa in QColor
            
            self.events_table.setItem(row, 3, impact_item)


class SignalsSummaryWidget(QWidget):
    """Widget per visualizzare un riepilogo dei segnali di trading."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget riepilogo segnali.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.signals = []  # Segnali di trading
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title = QLabel("Segnali di Trading")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        
        # Tabella segnali
        self.signals_table = QTableWidget()
        self.signals_table.setColumnCount(4)
        self.signals_table.setHorizontalHeaderLabels(["Simbolo", "Timeframe", "Azione", "Confidenza"])
        self.signals_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.signals_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self.signals_table)
        
        # Aggiorna i dati iniziali
        self.update_signals([])
    
    def update_signals(self, signals: List[Dict[str, Any]]):
        """
        Aggiorna i segnali di trading.
        
        Args:
            signals: Lista di dizionari con segnali di trading
        """
        self.signals = signals
        
        # Pulisci la tabella
        self.signals_table.setRowCount(0)
        
        if not signals:
            # Aggiungi alcune righe di esempio
            example_signals = [
                {
                    "symbol": "EURUSD",
                    "timeframe": "H1",
                    "action": "buy",
                    "confidence": 0.87
                },
                {
                    "symbol": "AAPL",
                    "timeframe": "D1",
                    "action": "sell",
                    "confidence": 0.92
                },
                {
                    "symbol": "BTCUSD",
                    "timeframe": "H4",
                    "action": "wait",
                    "confidence": 0.65
                }
            ]
            signals = example_signals
        
        # Aggiungi righe alla tabella
        for row, signal in enumerate(signals):
            self.signals_table.insertRow(row)
            
            # Simbolo
            symbol_item = QTableWidgetItem(signal["symbol"])
            self.signals_table.setItem(row, 0, symbol_item)
            
            # Timeframe
            timeframe_item = QTableWidgetItem(signal["timeframe"])
            self.signals_table.setItem(row, 1, timeframe_item)
            
            # Azione
            action = signal["action"]
            action_text = {
                "buy": "ACQUISTA",
                "sell": "VENDI",
                "wait": "ATTENDI"
            }.get(action, "")
            
            action_item = QTableWidgetItem(action_text)
            
            # Imposta colore in base all'azione
            if action == "buy":
                action_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif action == "sell":
                action_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.signals_table.setItem(row, 2, action_item)
            
            # Confidenza
            confidence = signal["confidence"]
            confidence_item = QTableWidgetItem(f"{confidence*100:.1f}%")
            
            # Imposta colore in base alla confidenza
            if confidence >= 0.8:
                confidence_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif confidence >= 0.6:
                confidence_item.setForeground(QColor(style_manager.colors.WARNING))  # Converto la stringa in QColor
            else:
                confidence_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.signals_table.setItem(row, 3, confidence_item)


class PerformanceSummaryWidget(QWidget):
    """Widget per visualizzare un riepilogo delle performance di trading."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget riepilogo performance.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.performance = {}  # Dati di performance
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Titolo
        title = QLabel("Performance di Trading")
        font = title.font()
        font.setBold(True)
        font.setPointSize(14)
        title.setFont(font)
        layout.addWidget(title)
        
        # Metriche principali
        metrics_grid = QGridLayout()
        
        # Rendimento totale
        metrics_grid.addWidget(QLabel("Rendimento:"), 0, 0)
        self.return_label = QLabel("+12.5%")
        self.return_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}; font-weight: bold; font-size: 14pt;")
        metrics_grid.addWidget(self.return_label, 0, 1)
        
        # Win Rate
        metrics_grid.addWidget(QLabel("Win Rate:"), 1, 0)
        self.winrate_label = QLabel("68.5%")
        self.winrate_label.setStyleSheet(f"font-weight: bold;")
        metrics_grid.addWidget(self.winrate_label, 1, 1)
        
        # Profit Factor
        metrics_grid.addWidget(QLabel("Profit Factor:"), 2, 0)
        self.pf_label = QLabel("2.34")
        self.pf_label.setStyleSheet(f"font-weight: bold;")
        metrics_grid.addWidget(self.pf_label, 2, 1)
        
        # Trades
        metrics_grid.addWidget(QLabel("Trades:"), 3, 0)
        self.trades_label = QLabel("124")
        metrics_grid.addWidget(self.trades_label, 3, 1)
        
        layout.addLayout(metrics_grid)
        
        # Separatore
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Tabella performance per strategia
        strat_label = QLabel("Performance per Strategia")
        strat_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(strat_label)
        
        self.strategies_table = QTableWidget()
        self.strategies_table.setColumnCount(3)
        self.strategies_table.setHorizontalHeaderLabels(["Strategia", "Rendimento", "Trades"])
        self.strategies_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.strategies_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self.strategies_table)
        
        # Aggiorna i dati iniziali
        self.update_performance({})
    
    def update_performance(self, performance: Dict[str, Any]):
        """
        Aggiorna i dati di performance.
        
        Args:
            performance: Dizionario con dati di performance
        """
        self.performance = performance
        
        if not performance:
            # Usa valori di esempio
            performance = {
                "total_return": 12.5,
                "win_rate": 68.5,
                "profit_factor": 2.34,
                "total_trades": 124,
                "strategies": [
                    {"name": "Moving Average Crossover", "return": 14.2, "trades": 45},
                    {"name": "RSI", "return": 8.7, "trades": 32},
                    {"name": "Bollinger Bands", "return": -2.3, "trades": 28}
                ]
            }
        
        # Aggiorna le etichette
        self.return_label.setText(f"{performance.get('total_return', 0):+.2f}%")
        if performance.get('total_return', 0) >= 0:
            self.return_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}; font-weight: bold; font-size: 14pt;")
        else:
            self.return_label.setStyleSheet(f"color: {style_manager.colors.ERROR}; font-weight: bold; font-size: 14pt;")
        
        self.winrate_label.setText(f"{performance.get('win_rate', 0):.1f}%")
        self.pf_label.setText(f"{performance.get('profit_factor', 0):.2f}")
        self.trades_label.setText(str(performance.get('total_trades', 0)))
        
        # Aggiorna la tabella delle strategie
        self.strategies_table.setRowCount(0)
        
        for row, strategy in enumerate(performance.get('strategies', [])):
            self.strategies_table.insertRow(row)
            
            # Nome strategia
            name_item = QTableWidgetItem(strategy["name"])
            self.strategies_table.setItem(row, 0, name_item)
            
            # Rendimento
            ret = strategy["return"]
            ret_item = QTableWidgetItem(f"{ret:+.2f}%")
            
            # Imposta colore
            if ret > 0:
                ret_item.setForeground(QColor(style_manager.colors.SUCCESS))  # Converto la stringa in QColor
            elif ret < 0:
                ret_item.setForeground(QColor(style_manager.colors.ERROR))  # Converto la stringa in QColor
            
            self.strategies_table.setItem(row, 1, ret_item)
            
            # Numero di trade
            trades_item = QTableWidgetItem(str(strategy["trades"]))
            self.strategies_table.setItem(row, 2, trades_item)


class DashboardWidget(QWidget):
    """Widget principale della dashboard che integra tutti i widget."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget dashboard.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
        
        # Timer per aggiornamento periodico
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_data)
        self.update_timer.start(60000)  # Aggiorna ogni minuto
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Intestazione
        header_layout = QHBoxLayout()
        
        title = QLabel("Dashboard")
        font = title.font()
        font.setBold(True)
        font.setPointSize(16)
        title.setFont(font)
        header_layout.addWidget(title)
        
        # Data e ora corrente
        self.date_label = QLabel(datetime.now().strftime("%d/%m/%Y %H:%M"))
        header_layout.addStretch()
        header_layout.addWidget(self.date_label)
        
        layout.addLayout(header_layout)
        
        # Separatore
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)
        
        # Contenuto principale
        # Usiamo un widget scrollabile
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Widget contenitore
        container = QWidget()
        container_layout = QGridLayout(container)
        
        # Riepilogo mercato
        self.market_summary = MarketSummaryWidget()
        container_layout.addWidget(self.market_summary, 0, 0)
        
        # Watchlist
        self.watchlist = WatchlistWidget()
        container_layout.addWidget(self.watchlist, 0, 1)
        
        # Mini grafici
        self.mini_charts = MiniChartWidget()
        container_layout.addWidget(self.mini_charts, 1, 0, 1, 2)
        
        # Calendario economico
        self.calendar = CalendarWidget()
        container_layout.addWidget(self.calendar, 2, 0)
        
        # Segnali
        self.signals = SignalsSummaryWidget()
        container_layout.addWidget(self.signals, 2, 1)
        
        # Performance
        self.performance = PerformanceSummaryWidget()
        container_layout.addWidget(self.performance, 3, 0, 1, 2)
        
        scroll_area.setWidget(container)
        layout.addWidget(scroll_area)
        
        self.setLayout(layout)
        
        # Connetti segnali
        self.watchlist.symbolSelected.connect(self.on_symbol_selected)
        
        # Aggiorna i dati iniziali
        self.update_data()
    
    def update_data(self):
        """Aggiorna tutti i dati della dashboard."""
        # In un'implementazione reale, qui dovresti caricare i dati dal backend
        
        # Aggiorna data e ora
        self.date_label.setText(datetime.now().strftime("%d/%m/%Y %H:%M"))
        
        # Aggiorna i widget
        # (usando dati di esempio, già gestiti nei widget stessi)
        self.market_summary.update_data([])
        self.watchlist.update_watchlist([])
        self.calendar.update_events([])
        self.signals.update_signals([])
        self.performance.update_performance({})
    
    def on_symbol_selected(self, symbol: str):
        """
        Gestisce la selezione di un simbolo nella watchlist.
        
        Args:
            symbol: Simbolo selezionato
        """
        # In un'implementazione reale, qui dovresti comunicare con altri componenti
        # Per ora, mostriamo solo un messaggio nella console
        app_logger.info(f"Simbolo selezionato: {symbol}")