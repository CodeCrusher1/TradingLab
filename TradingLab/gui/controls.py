# Widget di controllo

"""
Widget di controllo personalizzati per TradingLab.
Questo modulo fornisce controlli e widget riutilizzabili per l'interfaccia utente.
"""
from PyQt6.QtWidgets import (
    QWidget, QComboBox, QPushButton, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QLineEdit, QGridLayout, QCheckBox, QSpinBox, QDoubleSpinBox,
    QDateEdit, QToolButton, QSizePolicy, QScrollArea, QGroupBox
)
from PyQt6.QtCore import Qt, QDate, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QPixmap, QFont

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta
import os

from ..config import SYMBOLS, TIMEFRAMES
from ..config.symbols import get_symbol, get_timeframe, Symbol, Timeframe
from ..utils import app_logger
from .styles import style_manager, Theme


class SymbolSelector(QWidget):
    """
    Widget per la selezione di simboli di trading.
    Fornisce un combobox con tutti i simboli disponibili, raggruppati per asset class.
    """
    
    # Segnale emesso quando cambia il simbolo selezionato
    symbolChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di simboli.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Crea combobox per la selezione simboli
        self.symbolCombo = QComboBox()
        self.symbolCombo.setMinimumWidth(150)
        self.symbolCombo.setMaximumWidth(250)
        
        # Popola il combobox con i simboli disponibili
        self._populate_symbols()
        
        # Connetti segnale
        self.symbolCombo.currentTextChanged.connect(self._on_symbol_changed)
        
        # Aggiungi il combobox al layout
        layout.addWidget(QLabel("Simbolo:"))
        layout.addWidget(self.symbolCombo)
        
        self.setLayout(layout)
    
    def _populate_symbols(self):
        """Popola il combobox con i simboli disponibili, raggruppati per asset class."""
        self.symbolCombo.clear()
        
        # Raggruppa simboli per asset class
        symbol_groups = {}
        for name, symbol_obj in SYMBOLS.items():
            asset_class = symbol_obj.asset_class.value
            if asset_class not in symbol_groups:
                symbol_groups[asset_class] = []
            symbol_groups[asset_class].append(name)
        
        # Aggiungi al combobox
        for asset_class, symbols in symbol_groups.items():
            self.symbolCombo.addItem(f"---- {asset_class} ----")
            idx = self.symbolCombo.count() - 1
            self.symbolCombo.setItemData(idx, None, Qt.ItemDataRole.UserRole)  # Non selezionabile
            
            for symbol in sorted(symbols):
                self.symbolCombo.addItem(symbol)
                idx = self.symbolCombo.count() - 1
                self.symbolCombo.setItemData(idx, symbol, Qt.ItemDataRole.UserRole)
        
        # Seleziona il primo simbolo valido
        for i in range(self.symbolCombo.count()):
            if self.symbolCombo.itemData(i, Qt.ItemDataRole.UserRole) is not None:
                self.symbolCombo.setCurrentIndex(i)
                break
    
    def _on_symbol_changed(self, text):
        """
        Gestisce il cambio di simbolo selezionato.
        
        Args:
            text: Testo selezionato
        """
        # Ignora se il testo corrisponde a un'intestazione
        current_idx = self.symbolCombo.currentIndex()
        if self.symbolCombo.itemData(current_idx, Qt.ItemDataRole.UserRole) is None:
            # Se è un'intestazione, trova il prossimo simbolo valido
            for i in range(current_idx + 1, self.symbolCombo.count()):
                if self.symbolCombo.itemData(i, Qt.ItemDataRole.UserRole) is not None:
                    self.symbolCombo.setCurrentIndex(i)
                    return
            
            # Se non ce ne sono dopo, cerca prima
            for i in range(current_idx):
                if self.symbolCombo.itemData(i, Qt.ItemDataRole.UserRole) is not None:
                    self.symbolCombo.setCurrentIndex(i)
                    return
        else:
            # Emetti il segnale con il simbolo selezionato
            self.symbolChanged.emit(text)
    
    def get_selected_symbol(self) -> str:
        """
        Ottiene il simbolo attualmente selezionato.
        
        Returns:
            Nome del simbolo selezionato
        """
        current_idx = self.symbolCombo.currentIndex()
        return self.symbolCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
    
    def set_selected_symbol(self, symbol: str) -> bool:
        """
        Imposta il simbolo selezionato.
        
        Args:
            symbol: Nome del simbolo da selezionare
        
        Returns:
            True se il simbolo è stato trovato e selezionato
        """
        for i in range(self.symbolCombo.count()):
            if self.symbolCombo.itemData(i, Qt.ItemDataRole.UserRole) == symbol:
                self.symbolCombo.setCurrentIndex(i)
                return True
        return False


class TimeframeSelector(QWidget):
    """Widget per la selezione di timeframe."""
    
    # Segnale emesso quando cambia il timeframe selezionato
    timeframeChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di timeframe.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Crea combobox per la selezione timeframe
        self.timeframeCombo = QComboBox()
        self.timeframeCombo.setMinimumWidth(120)
        self.timeframeCombo.setMaximumWidth(200)
        
        # Popola il combobox con i timeframe disponibili
        self._populate_timeframes()
        
        # Connetti segnale
        self.timeframeCombo.currentTextChanged.connect(self._on_timeframe_changed)
        
        # Aggiungi il combobox al layout
        layout.addWidget(QLabel("Timeframe:"))
        layout.addWidget(self.timeframeCombo)
        
        self.setLayout(layout)
    
    def _populate_timeframes(self):
        """Popola il combobox con i timeframe disponibili."""
        self.timeframeCombo.clear()
        
        # Ordina i timeframe per minuti
        timeframes_sorted = sorted(
            TIMEFRAMES.items(), 
            key=lambda x: x[1].minutes
        )
        
        for name, tf in timeframes_sorted:
            self.timeframeCombo.addItem(f"{tf.description} ({tf.code})")
            idx = self.timeframeCombo.count() - 1
            self.timeframeCombo.setItemData(idx, name, Qt.ItemDataRole.UserRole)
        
        # Seleziona il primo timeframe
        if self.timeframeCombo.count() > 0:
            self.timeframeCombo.setCurrentIndex(0)
    
    def _on_timeframe_changed(self, text):
        """
        Gestisce il cambio di timeframe selezionato.
        
        Args:
            text: Testo selezionato
        """
        current_idx = self.timeframeCombo.currentIndex()
        timeframe = self.timeframeCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
        if timeframe:
            self.timeframeChanged.emit(timeframe)
    
    def get_selected_timeframe(self) -> str:
        """
        Ottiene il timeframe attualmente selezionato.
        
        Returns:
            Nome del timeframe selezionato
        """
        current_idx = self.timeframeCombo.currentIndex()
        return self.timeframeCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
    
    def set_selected_timeframe(self, timeframe: str) -> bool:
        """
        Imposta il timeframe selezionato.
        
        Args:
            timeframe: Nome del timeframe da selezionare
        
        Returns:
            True se il timeframe è stato trovato e selezionato
        """
        for i in range(self.timeframeCombo.count()):
            if self.timeframeCombo.itemData(i, Qt.ItemDataRole.UserRole) == timeframe:
                self.timeframeCombo.setCurrentIndex(i)
                return True
        return False


class DateRangeSelector(QWidget):
    """Widget per la selezione di un intervallo di date."""
    
    # Segnale emesso quando cambia l'intervallo di date
    dateRangeChanged = pyqtSignal(datetime, datetime)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di intervallo date.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Layout per le date
        date_layout = QHBoxLayout()
        
        # Data di inizio
        start_layout = QVBoxLayout()
        start_layout.addWidget(QLabel("Data inizio:"))
        self.startDateEdit = QDateEdit()
        self.startDateEdit.setCalendarPopup(True)
        self.startDateEdit.setDate(QDate.currentDate().addMonths(-3))
        start_layout.addWidget(self.startDateEdit)
        
        # Data di fine
        end_layout = QVBoxLayout()
        end_layout.addWidget(QLabel("Data fine:"))
        self.endDateEdit = QDateEdit()
        self.endDateEdit.setCalendarPopup(True)
        self.endDateEdit.setDate(QDate.currentDate())
        end_layout.addWidget(self.endDateEdit)
        
        # Aggiungi al layout principale
        date_layout.addLayout(start_layout)
        date_layout.addLayout(end_layout)
        
        layout.addLayout(date_layout)
        
        # Pulsanti per intervalli predefiniti
        preset_layout = QHBoxLayout()
        
        self.btn1M = QPushButton("1M")
        self.btn3M = QPushButton("3M")
        self.btn6M = QPushButton("6M")
        self.btn1Y = QPushButton("1Y")
        self.btnYTD = QPushButton("YTD")
        
        self.btn1M.setMaximumWidth(40)
        self.btn3M.setMaximumWidth(40)
        self.btn6M.setMaximumWidth(40)
        self.btn1Y.setMaximumWidth(40)
        self.btnYTD.setMaximumWidth(40)
        
        preset_layout.addWidget(self.btn1M)
        preset_layout.addWidget(self.btn3M)
        preset_layout.addWidget(self.btn6M)
        preset_layout.addWidget(self.btn1Y)
        preset_layout.addWidget(self.btnYTD)
        
        layout.addLayout(preset_layout)
        
        # Connetti segnali
        self.startDateEdit.dateChanged.connect(self._on_date_changed)
        self.endDateEdit.dateChanged.connect(self._on_date_changed)
        
        self.btn1M.clicked.connect(lambda: self.set_preset_range(months=1))
        self.btn3M.clicked.connect(lambda: self.set_preset_range(months=3))
        self.btn6M.clicked.connect(lambda: self.set_preset_range(months=6))
        self.btn1Y.clicked.connect(lambda: self.set_preset_range(years=1))
        self.btnYTD.clicked.connect(self.set_ytd_range)
        
        self.setLayout(layout)
    
    def _on_date_changed(self):
        """Gestisce il cambio di date selezionate."""
        start_date = self.startDateEdit.date().toPyDate()
        end_date = self.endDateEdit.date().toPyDate()
        
        # Verifica che la data di inizio sia precedente alla data di fine
        if start_date > end_date:
            # Imposta la data di fine uguale alla data di inizio
            self.endDateEdit.setDate(self.startDateEdit.date())
            end_date = start_date
        
        # Converte le date in datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # Emetti il segnale
        self.dateRangeChanged.emit(start_datetime, end_datetime)
    
    def set_preset_range(self, days=0, months=0, years=0):
        """
        Imposta un intervallo di date predefinito.
        
        Args:
            days: Numero di giorni indietro
            months: Numero di mesi indietro
            years: Numero di anni indietro
        """
        end_date = QDate.currentDate()
        start_date = QDate(end_date)
        
        if days > 0:
            start_date = start_date.addDays(-days)
        if months > 0:
            start_date = start_date.addMonths(-months)
        if years > 0:
            start_date = start_date.addYears(-years)
        
        self.startDateEdit.setDate(start_date)
        self.endDateEdit.setDate(end_date)
    
    def set_ytd_range(self):
        """Imposta l'intervallo 'Year to Date'."""
        end_date = QDate.currentDate()
        start_date = QDate(end_date.year(), 1, 1)
        
        self.startDateEdit.setDate(start_date)
        self.endDateEdit.setDate(end_date)
    
    def get_date_range(self) -> Tuple[datetime, datetime]:
        """
        Ottiene l'intervallo di date attualmente selezionato.
        
        Returns:
            Tupla (start_date, end_date)
        """
        start_date = self.startDateEdit.date().toPyDate()
        end_date = self.endDateEdit.date().toPyDate()
        
        # Converte le date in datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        return start_datetime, end_datetime
    
    def set_date_range(self, start_date: datetime, end_date: datetime):
        """
        Imposta l'intervallo di date.
        
        Args:
            start_date: Data di inizio
            end_date: Data di fine
        """
        self.startDateEdit.setDate(QDate(start_date.year, start_date.month, start_date.day))
        self.endDateEdit.setDate(QDate(end_date.year, end_date.month, end_date.day))


class IndicatorSelector(QWidget):
    """
    Widget per la selezione di indicatori tecnici.
    Permette all'utente di selezionare quali indicatori visualizzare nei grafici.
    """
    
    # Segnale emesso quando cambiano gli indicatori selezionati
    indicatorsChanged = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di indicatori.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.indicators = {
            "trend": {
                "ema": True,
                "sma": False,
                "supertrend": False,
                "parabolic_sar": False,
                "adx": False,
                "keltner": False
            },
            "momentum": {
                "rsi": True,
                "macd": False,
                "stochastic": False,
                "awesome_oscillator": False,
                "williams_r": False
            },
            "volume": {
                "volume": True,
                "on_balance_volume": False,
                "accumulation_distribution": False,
                "volume_profile": False
            },
            "volatility": {
                "bollinger_bands": False,
                "atr": True
            },
            "patterns": {
                "candle_patterns": False,
                "chart_patterns": False,
                "harmonic_patterns": False
            },
            "market_structure": {
                "support_resistance": True,
                "fair_value_gaps": False,
                "order_blocks": False,
                "breaker_blocks": False,
                "liquidity_levels": False
            }
        }
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        # Layout esterno
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Creiamo un widget scrollabile per contenere tutti gli indicatori
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Widget contenitore per gli indicatori
        indicator_widget = QWidget()
        indicator_layout = QVBoxLayout(indicator_widget)
        
        # Crea sezioni per ciascuna categoria di indicatori
        for category, indicators in self.indicators.items():
            group_box = QGroupBox(category.capitalize())
            group_layout = QVBoxLayout()
            
            for indicator, checked in indicators.items():
                # Trasforma il nome dell'indicatore (es. "ema" -> "EMA")
                display_name = indicator.replace('_', ' ').title()
                
                checkbox = QCheckBox(display_name)
                checkbox.setChecked(checked)
                checkbox.setObjectName(f"{category}_{indicator}")
                checkbox.stateChanged.connect(self._on_indicator_changed)
                
                group_layout.addWidget(checkbox)
            
            group_box.setLayout(group_layout)
            indicator_layout.addWidget(group_box)
        
        # Aggiungi uno stretcher alla fine
        indicator_layout.addStretch()
        
        # Imposta il widget all'area scrollabile
        scroll_area.setWidget(indicator_widget)
        
        # Aggiungi l'area scrollabile al layout principale
        main_layout.addWidget(scroll_area)
    
    def _on_indicator_changed(self, state):
        """
        Gestisce il cambio di stato di un indicatore.
        
        Args:
            state: Stato del checkbox
        """
        # Ottieni il checkbox che ha emesso il segnale
        checkbox = self.sender()
        if checkbox:
            # Estrai categoria e nome indicatore dall'objectName
            category, indicator = checkbox.objectName().split('_', 1)
            
            # Aggiorna lo stato dell'indicatore
            self.indicators[category][indicator] = checkbox.isChecked()
            
            # Emetti il segnale con tutti gli indicatori
            self.indicatorsChanged.emit(self.indicators)
    
    def get_selected_indicators(self) -> Dict[str, Dict[str, bool]]:
        """
        Ottiene gli indicatori attualmente selezionati.
        
        Returns:
            Dizionario con gli indicatori selezionati
        """
        return self.indicators
    
    def set_indicator(self, category: str, indicator: str, checked: bool):
        """
        Imposta lo stato di un indicatore specifico.
        
        Args:
            category: Categoria dell'indicatore
            indicator: Nome dell'indicatore
            checked: Se l'indicatore deve essere selezionato
        """
        if category in self.indicators and indicator in self.indicators[category]:
            self.indicators[category][indicator] = checked
            
            # Aggiorna anche l'UI
            checkbox = self.findChild(QCheckBox, f"{category}_{indicator}")
            if checkbox:
                checkbox.setChecked(checked)


class ModelSelector(QWidget):
    """
    Widget per la selezione di modelli di previsione.
    Permette all'utente di selezionare quale modello utilizzare per le previsioni.
    """
    
    # Segnale emesso quando cambia il modello selezionato
    modelChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di modelli.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.models = []  # Lista di modelli disponibili
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        layout.addWidget(QLabel("Modello di previsione:"))
        
        # Combobox per la selezione del modello
        self.modelCombo = QComboBox()
        
        # Aggiungi opzione per analisi tecnica standard
        self.modelCombo.addItem("Analisi Tecnica")
        self.modelCombo.setItemData(0, "technical_analysis", Qt.ItemDataRole.UserRole)
        
        # Connetti segnali
        self.modelCombo.currentIndexChanged.connect(self._on_model_changed)
        
        # Aggiungi al layout
        layout.addWidget(self.modelCombo)
        
        # Pulsante per caricare i modelli disponibili
        self.loadModelsBtn = QPushButton("Carica modelli")
        self.loadModelsBtn.clicked.connect(self._load_available_models)
        layout.addWidget(self.loadModelsBtn)
        
        self.setLayout(layout)
    
    def _load_available_models(self):
        """Carica l'elenco di modelli disponibili."""
        # In un'implementazione reale, questo metodo dovrebbe caricare i modelli
        # dal registro dei modelli o da una directory specifica.
        # Per ora, aggiungeremo solo alcuni modelli di esempio.
        
        # Mantieni l'opzione "Analisi Tecnica"
        current_model = self.get_selected_model()
        
        self.modelCombo.clear()
        
        # Aggiungi opzione per analisi tecnica standard
        self.modelCombo.addItem("Analisi Tecnica")
        self.modelCombo.setItemData(0, "technical_analysis", Qt.ItemDataRole.UserRole)
        
        # Modelli di esempio
        example_models = [
            {"name": "LSTM - Trend Follower", "id": "ta_lstm_trend", "description": "LSTM con indicatori tecnici"},
            {"name": "Transformer - Swing", "id": "transformer_swing", "description": "Modello Transformer per swing trading"},
            {"name": "Ensemble - Multi-Timeframe", "id": "ensemble_mtf", "description": "Modello ensemble con analisi multi-timeframe"}
        ]
        
        for model in example_models:
            self.modelCombo.addItem(model["name"])
            idx = self.modelCombo.count() - 1
            self.modelCombo.setItemData(idx, model["id"], Qt.ItemDataRole.UserRole)
            self.modelCombo.setItemData(idx, model["description"], Qt.ItemDataRole.ToolTipRole)
        
        # Ripristina il modello selezionato se possibile
        if current_model:
            self.set_selected_model(current_model)
        
        # Memorizza i modelli
        self.models = example_models
    
    def _on_model_changed(self, index):
        """
        Gestisce il cambio di modello selezionato.
        
        Args:
            index: Indice selezionato nella combobox
        """
        model_id = self.modelCombo.itemData(index, Qt.ItemDataRole.UserRole)
        if model_id:
            self.modelChanged.emit(model_id)
    
    def get_selected_model(self) -> str:
        """
        Ottiene l'ID del modello attualmente selezionato.
        
        Returns:
            ID del modello
        """
        current_idx = self.modelCombo.currentIndex()
        if current_idx >= 0:
            return self.modelCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
        return None
    
    def set_selected_model(self, model_id: str) -> bool:
        """
        Imposta il modello selezionato.
        
        Args:
            model_id: ID del modello da selezionare
        
        Returns:
            True se il modello è stato trovato e selezionato
        """
        for i in range(self.modelCombo.count()):
            if self.modelCombo.itemData(i, Qt.ItemDataRole.UserRole) == model_id:
                self.modelCombo.setCurrentIndex(i)
                return True
        return False


class StrategySelector(QWidget):
    """
    Widget per la selezione di strategie di trading.
    Permette all'utente di selezionare quale strategia utilizzare per il backtest.
    """
    
    # Segnale emesso quando cambia la strategia selezionata
    strategyChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il selettore di strategie.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.strategies = {}  # Dizionario di strategie disponibili
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        layout.addWidget(QLabel("Strategia di trading:"))
        
        # Combobox per la selezione della strategia
        self.strategyCombo = QComboBox()
        
        # Connetti segnali
        self.strategyCombo.currentTextChanged.connect(self._on_strategy_changed)
        
        # Aggiungi al layout
        layout.addWidget(self.strategyCombo)
        
        # Sezione parametri
        self.paramsGroup = QGroupBox("Parametri")
        self.paramsLayout = QGridLayout()
        self.paramsGroup.setLayout(self.paramsLayout)
        
        layout.addWidget(self.paramsGroup)
        
        self.setLayout(layout)
        
        # Carica strategie disponibili
        self._load_available_strategies()
    
    def _load_available_strategies(self):
        """Carica l'elenco di strategie disponibili."""
        from ..prediction.strategies import StrategyFactory
        
        # Ottieni le strategie disponibili
        available_strategies = StrategyFactory.get_available_strategies()
        
        self.strategyCombo.clear()
        
        # Aggiungi le strategie al combobox
        for strategy_id, description in available_strategies.items():
            # Converti ID in nome visualizzabile (es. "moving_average_crossover" -> "Moving Average Crossover")
            display_name = strategy_id.replace('_', ' ').title()
            
            self.strategyCombo.addItem(display_name)
            idx = self.strategyCombo.count() - 1
            self.strategyCombo.setItemData(idx, strategy_id, Qt.ItemDataRole.UserRole)
            self.strategyCombo.setItemData(idx, description, Qt.ItemDataRole.ToolTipRole)
            
            # Memorizza strategia
            self.strategies[strategy_id] = {"name": display_name, "description": description}
        
        # Seleziona la prima strategia
        if self.strategyCombo.count() > 0:
            self.strategyCombo.setCurrentIndex(0)
    
    def _on_strategy_changed(self, text):
        """
        Gestisce il cambio di strategia selezionata.
        
        Args:
            text: Testo selezionato
        """
        current_idx = self.strategyCombo.currentIndex()
        strategy_id = self.strategyCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
        
        if strategy_id:
            # Aggiorna i parametri
            self._update_strategy_params(strategy_id)
            
            # Emetti il segnale
            self.strategyChanged.emit(strategy_id)
    
    def _update_strategy_params(self, strategy_id: str):
        """
        Aggiorna i widget dei parametri per la strategia selezionata.
        
        Args:
            strategy_id: ID della strategia
        """
        # Pulisci i parametri esistenti
        for i in reversed(range(self.paramsLayout.count())): 
            self.paramsLayout.itemAt(i).widget().setParent(None)
        
        # Crea i parametri per la strategia selezionata
        # Questi sono solo esempi, in un'implementazione reale dovrebbero
        # essere presi dalla strategia stessa
        
        if strategy_id == "moving_average_crossover":
            # Parametri per MA Crossover
            self.paramsLayout.addWidget(QLabel("Periodo veloce:"), 0, 0)
            fast_spinbox = QSpinBox()
            fast_spinbox.setMinimum(5)
            fast_spinbox.setMaximum(50)
            fast_spinbox.setValue(20)
            fast_spinbox.setObjectName("fast_period")
            self.paramsLayout.addWidget(fast_spinbox, 0, 1)
            
            self.paramsLayout.addWidget(QLabel("Periodo lento:"), 1, 0)
            slow_spinbox = QSpinBox()
            slow_spinbox.setMinimum(20)
            slow_spinbox.setMaximum(200)
            slow_spinbox.setValue(50)
            slow_spinbox.setObjectName("slow_period")
            self.paramsLayout.addWidget(slow_spinbox, 1, 1)
            
            self.paramsLayout.addWidget(QLabel("Tipo MA:"), 2, 0)
            ma_combo = QComboBox()
            ma_combo.addItem("EMA")
            ma_combo.addItem("SMA")
            ma_combo.setObjectName("ma_type")
            self.paramsLayout.addWidget(ma_combo, 2, 1)
            
        elif strategy_id == "rsi":
            # Parametri per RSI
            self.paramsLayout.addWidget(QLabel("Periodo RSI:"), 0, 0)
            rsi_spinbox = QSpinBox()
            rsi_spinbox.setMinimum(5)
            rsi_spinbox.setMaximum(50)
            rsi_spinbox.setValue(14)
            rsi_spinbox.setObjectName("rsi_period")
            self.paramsLayout.addWidget(rsi_spinbox, 0, 1)
            
            self.paramsLayout.addWidget(QLabel("Ipercomprato:"), 1, 0)
            overbought_spinbox = QSpinBox()
            overbought_spinbox.setMinimum(50)
            overbought_spinbox.setMaximum(90)
            overbought_spinbox.setValue(70)
            overbought_spinbox.setObjectName("overbought")
            self.paramsLayout.addWidget(overbought_spinbox, 1, 1)
            
            self.paramsLayout.addWidget(QLabel("Ipervenduto:"), 2, 0)
            oversold_spinbox = QSpinBox()
            oversold_spinbox.setMinimum(10)
            oversold_spinbox.setMaximum(50)
            oversold_spinbox.setValue(30)
            oversold_spinbox.setObjectName("oversold")
            self.paramsLayout.addWidget(oversold_spinbox, 2, 1)
            
        elif strategy_id == "supertrend":
            # Parametri per Supertrend
            self.paramsLayout.addWidget(QLabel("Periodo ATR:"), 0, 0)
            atr_spinbox = QSpinBox()
            atr_spinbox.setMinimum(5)
            atr_spinbox.setMaximum(30)
            atr_spinbox.setValue(10)
            atr_spinbox.setObjectName("atr_period")
            self.paramsLayout.addWidget(atr_spinbox, 0, 1)
            
            self.paramsLayout.addWidget(QLabel("Moltiplicatore:"), 1, 0)
            mult_spinbox = QDoubleSpinBox()
            mult_spinbox.setMinimum(1.0)
            mult_spinbox.setMaximum(5.0)
            mult_spinbox.setSingleStep(0.1)
            mult_spinbox.setValue(3.0)
            mult_spinbox.setObjectName("multiplier")
            self.paramsLayout.addWidget(mult_spinbox, 1, 1)
        
        # Altri tipi di strategia...
        
        else:
            # Caso generico
            self.paramsLayout.addWidget(QLabel("Parametri non disponibili per questa strategia"), 0, 0, 1, 2)
    
    def get_selected_strategy(self) -> str:
        """
        Ottiene l'ID della strategia attualmente selezionata.
        
        Returns:
            ID della strategia
        """
        current_idx = self.strategyCombo.currentIndex()
        if current_idx >= 0:
            return self.strategyCombo.itemData(current_idx, Qt.ItemDataRole.UserRole)
        return None
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Ottiene i parametri impostati per la strategia selezionata.
        
        Returns:
            Dizionario con i parametri
        """
        params = {}
        
        # Raccogli i valori da tutti i widget di input nella sezione parametri
        for i in range(self.paramsLayout.count()):
            widget = self.paramsLayout.itemAt(i).widget()
            
            # Controlla se è un widget di input e ha un objectName
            if hasattr(widget, 'objectName') and widget.objectName():
                param_name = widget.objectName()
                
                # Raccogli il valore in base al tipo di widget
                if isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
                    params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    params[param_name] = widget.currentText().lower()
                elif isinstance(widget, QCheckBox):
                    params[param_name] = widget.isChecked()
                elif isinstance(widget, QLineEdit):
                    params[param_name] = widget.text()
        
        return params
    
    def set_selected_strategy(self, strategy_id: str) -> bool:
        """
        Imposta la strategia selezionata.
        
        Args:
            strategy_id: ID della strategia da selezionare
        
        Returns:
            True se la strategia è stata trovata e selezionata
        """
        for i in range(self.strategyCombo.count()):
            if self.strategyCombo.itemData(i, Qt.ItemDataRole.UserRole) == strategy_id:
                self.strategyCombo.setCurrentIndex(i)
                return True
        return False


class TradeSignalView(QWidget):
    """Widget per visualizzare segnali di trading."""
    
    def __init__(self, parent=None):
        """
        Inizializza il visualizzatore di segnali.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.signals = []  # Lista di segnali
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Intestazione
        header = QLabel("Segnali di Trading")
        font = header.font()
        font.setBold(True)
        font.setPointSize(12)
        header.setFont(font)
        
        layout.addWidget(header)
        
        # Frame per i segnali
        signals_frame = QFrame()
        signals_frame.setFrameShape(QFrame.Shape.StyledPanel)
        signals_frame.setStyleSheet(f"background-color: {style_manager.colors.CARD_BG}")
        
        self.signals_layout = QVBoxLayout(signals_frame)
        
        # Placeholder per quando non ci sono segnali
        self.placeholder = QLabel("Nessun segnale disponibile")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.signals_layout.addWidget(self.placeholder)
        
        # Aggiungi il frame al layout principale
        layout.addWidget(signals_frame)
        
        self.setLayout(layout)
    
    def update_signals(self, signals: List[Dict[str, Any]]):
        """
        Aggiorna la lista di segnali visualizzati.
        
        Args:
            signals: Lista di dizionari con i segnali
        """
        # Pulisci i segnali esistenti
        for i in reversed(range(self.signals_layout.count())): 
            self.signals_layout.itemAt(i).widget().setParent(None)
        
        self.signals = signals
        
        if not signals:
            # Mostra placeholder se non ci sono segnali
            self.placeholder = QLabel("Nessun segnale disponibile")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.signals_layout.addWidget(self.placeholder)
            return
        
        # Aggiungi i nuovi segnali
        for signal in signals:
            signal_widget = self._create_signal_widget(signal)
            self.signals_layout.addWidget(signal_widget)
        
        # Aggiungi uno stretcher alla fine
        self.signals_layout.addStretch()
    
    def _create_signal_widget(self, signal: Dict[str, Any]) -> QFrame:
        """
        Crea un widget per un singolo segnale.
        
        Args:
            signal: Dizionario con i dati del segnale
            
        Returns:
            Widget del segnale
        """
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        
        # Imposta lo stile in base al tipo di segnale
        if signal.get('action') == 'buy':
            frame.setStyleSheet(f"background-color: {style_manager.colors.SUCCESS}; color: white; border-radius: 4px;")
        elif signal.get('action') == 'sell':
            frame.setStyleSheet(f"background-color: {style_manager.colors.ERROR}; color: white; border-radius: 4px;")
        else:
            frame.setStyleSheet(f"background-color: {style_manager.colors.NEUTRAL}; color: white; border-radius: 4px;")
        
        layout = QVBoxLayout(frame)
        
        # Simbolo e timeframe
        header_layout = QHBoxLayout()
        symbol_label = QLabel(signal.get('symbol', 'Unknown'))
        font = symbol_label.font()
        font.setBold(True)
        symbol_label.setFont(font)
        
        timeframe_label = QLabel(signal.get('timeframe', ''))
        
        header_layout.addWidget(symbol_label)
        header_layout.addWidget(timeframe_label)
        header_layout.addStretch()
        
        # Data e ora
        if 'timestamp' in signal:
            date_time = signal['timestamp']
            if isinstance(date_time, str):
                try:
                    date_time = datetime.fromisoformat(date_time)
                except:
                    pass
            
            if isinstance(date_time, datetime):
                time_label = QLabel(date_time.strftime('%d/%m/%Y %H:%M'))
                header_layout.addWidget(time_label)
        
        layout.addLayout(header_layout)
        
        # Tipo di segnale
        action_text = {
            'buy': 'ACQUISTA',
            'sell': 'VENDI',
            'wait': 'ATTENDI'
        }.get(signal.get('action', ''), 'SCONOSCIUTO')
        
        action_label = QLabel(action_text)
        font = action_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        action_label.setFont(font)
        action_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(action_label)
        
        # Dettagli aggiuntivi
        details_layout = QGridLayout()
        
        row = 0
        if 'entry_price' in signal and signal['entry_price'] is not None:
            details_layout.addWidget(QLabel("Prezzo entrata:"), row, 0)
            details_layout.addWidget(QLabel(f"{signal['entry_price']:.2f}"), row, 1)
            row += 1
        
        if 'tp_price' in signal and signal['tp_price'] is not None:
            details_layout.addWidget(QLabel("Take Profit:"), row, 0)
            details_layout.addWidget(QLabel(f"{signal['tp_price']:.2f}"), row, 1)
            row += 1
        
        if 'sl_price' in signal and signal['sl_price'] is not None:
            details_layout.addWidget(QLabel("Stop Loss:"), row, 0)
            details_layout.addWidget(QLabel(f"{signal['sl_price']:.2f}"), row, 1)
            row += 1
        
        if 'risk_reward_ratio' in signal and signal['risk_reward_ratio'] is not None:
            details_layout.addWidget(QLabel("Risk/Reward:"), row, 0)
            details_layout.addWidget(QLabel(f"{signal['risk_reward_ratio']:.2f}"), row, 1)
            row += 1
        
        if 'confidence' in signal and signal['confidence'] is not None:
            details_layout.addWidget(QLabel("Confidenza:"), row, 0)
            details_layout.addWidget(QLabel(f"{signal['confidence']*100:.1f}%"), row, 1)
            row += 1
        
        if row > 0:
            layout.addLayout(details_layout)
        
        return frame


class PredictionResultView(QWidget):
    """Widget per visualizzare i risultati di una previsione."""
    
    def __init__(self, parent=None):
        """
        Inizializza il visualizzatore di previsioni.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.prediction = None  # Risultato della previsione
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Intestazione
        header = QLabel("Risultato Previsione")
        font = header.font()
        font.setBold(True)
        font.setPointSize(12)
        header.setFont(font)
        
        layout.addWidget(header)
        
        # Frame principale
        main_frame = QFrame()
        main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        main_frame.setStyleSheet(f"background-color: {style_manager.colors.CARD_BG}")
        
        self.main_layout = QVBoxLayout(main_frame)
        
        # Placeholder quando non ci sono previsioni
        self.placeholder = QLabel("Nessuna previsione disponibile")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.placeholder)
        
        # Aggiungi il frame al layout principale
        layout.addWidget(main_frame)
        
        self.setLayout(layout)
    
    def update_prediction(self, prediction: Optional[Dict[str, Any]]):
        """
        Aggiorna la previsione visualizzata.
        
        Args:
            prediction: Dizionario con i dati della previsione o None
        """
        # Pulisci i widget esistenti
        for i in reversed(range(self.main_layout.count())): 
            self.main_layout.itemAt(i).widget().setParent(None)
        
        self.prediction = prediction
        
        if not prediction:
            # Mostra placeholder se non ci sono previsioni
            self.placeholder = QLabel("Nessuna previsione disponibile")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_layout.addWidget(self.placeholder)
            return
        
        # Crea i widget per la previsione
        
        # Titolo
        symbol_label = QLabel(f"{prediction.get('symbol', 'Unknown')} - {prediction.get('timeframe', '')}")
        font = symbol_label.font()
        font.setBold(True)
        font.setPointSize(12)
        symbol_label.setFont(font)
        
        self.main_layout.addWidget(symbol_label)
        
        # Data previsione
        if 'timestamp' in prediction:
            date_time = prediction['timestamp']
            if isinstance(date_time, str):
                try:
                    date_time = datetime.fromisoformat(date_time)
                except:
                    pass
            
            if isinstance(date_time, datetime):
                time_label = QLabel(f"Data: {date_time.strftime('%d/%m/%Y')}")
                self.main_layout.addWidget(time_label)
        
        # Separatore
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)
        
        # Risultato previsione
        prediction_layout = QHBoxLayout()
        
        # Ottieni la direzione e l'azione
        direction = prediction.get('direction', 0)
        action = prediction.get('action', 'wait')
        
        # Crea etichetta per la direzione
        direction_text = ""
        if direction > 0:
            direction_text = "RIALZISTA"
            direction_color = style_manager.colors.SUCCESS
        elif direction < 0:
            direction_text = "RIBASSISTA"
            direction_color = style_manager.colors.ERROR
        else:
            direction_text = "NEUTRALE"
            direction_color = style_manager.colors.NEUTRAL
        
        direction_label = QLabel(direction_text)
        font = direction_label.font()
        font.setBold(True)
        font.setPointSize(14)
        direction_label.setFont(font)
        direction_label.setStyleSheet(f"color: {direction_color}")
        direction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        prediction_layout.addWidget(direction_label)
        self.main_layout.addLayout(prediction_layout)
        
        # Azione consigliata
        action_text = {
            'buy': 'ACQUISTA',
            'sell': 'VENDI',
            'wait': 'ATTENDI'
        }.get(action, 'SCONOSCIUTO')
        
        action_label = QLabel(f"Azione consigliata: {action_text}")
        font = action_label.font()
        font.setBold(True)
        self.main_layout.addWidget(action_label)
        
        # Dettagli previsione
        details_layout = QGridLayout()
        
        row = 0
        # Confidenza
        if 'confidence' in prediction and prediction['confidence'] is not None:
            details_layout.addWidget(QLabel("Confidenza:"), row, 0)
            confidence_value = prediction['confidence'] * 100
            confidence_label = QLabel(f"{confidence_value:.1f}%")
            
            # Colore in base al valore
            if confidence_value >= 75:
                confidence_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif confidence_value >= 50:
                confidence_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                confidence_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            details_layout.addWidget(confidence_label, row, 1)
            row += 1
        
        # Prezzo attuale/entrata
        if 'entry_price' in prediction and prediction['entry_price'] is not None:
            details_layout.addWidget(QLabel("Prezzo attuale:"), row, 0)
            details_layout.addWidget(QLabel(f"{prediction['entry_price']:.2f}"), row, 1)
            row += 1
        
        # Take Profit
        if 'tp_price' in prediction and prediction['tp_price'] is not None:
            details_layout.addWidget(QLabel("Take Profit:"), row, 0)
            details_layout.addWidget(QLabel(f"{prediction['tp_price']:.2f}"), row, 1)
            row += 1
        
        # Stop Loss
        if 'sl_price' in prediction and prediction['sl_price'] is not None:
            details_layout.addWidget(QLabel("Stop Loss:"), row, 0)
            details_layout.addWidget(QLabel(f"{prediction['sl_price']:.2f}"), row, 1)
            row += 1
        
        # Risk/Reward
        if 'risk_reward_ratio' in prediction and prediction['risk_reward_ratio'] is not None:
            details_layout.addWidget(QLabel("Risk/Reward:"), row, 0)
            risk_reward = prediction['risk_reward_ratio']
            risk_reward_label = QLabel(f"{risk_reward:.2f}")
            
            # Colore in base al valore
            if risk_reward >= 2.0:
                risk_reward_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif risk_reward >= 1.0:
                risk_reward_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                risk_reward_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            details_layout.addWidget(risk_reward_label, row, 1)
            row += 1
        
        if row > 0:
            self.main_layout.addLayout(details_layout)
        
        # Separatore
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        line2.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line2)
        
        # Informazioni sul modello
        if 'model_name' in prediction and prediction['model_name']:
            model_label = QLabel(f"Modello: {prediction['model_name']}")
            self.main_layout.addWidget(model_label)
            
            if 'model_version' in prediction and prediction['model_version']:
                version_label = QLabel(f"Versione: {prediction['model_version']}")
                self.main_layout.addWidget(version_label)
        
        # Distribuzione probabilità
        if 'probability_distribution' in prediction and prediction['probability_distribution']:
            prob_dist = prediction['probability_distribution']
            
            # Header
            prob_header = QLabel("Distribuzione di probabilità:")
            font = prob_header.font()
            font.setBold(True)
            prob_header.setFont(font)
            self.main_layout.addWidget(prob_header)
            
            # Valori
            prob_layout = QGridLayout()
            
            if 'up' in prob_dist:
                prob_layout.addWidget(QLabel("Rialzo:"), 0, 0)
                prob_layout.addWidget(QLabel(f"{prob_dist['up']*100:.1f}%"), 0, 1)
            
            if 'neutral' in prob_dist:
                prob_layout.addWidget(QLabel("Neutrale:"), 1, 0)
                prob_layout.addWidget(QLabel(f"{prob_dist['neutral']*100:.1f}%"), 1, 1)
            
            if 'down' in prob_dist:
                prob_layout.addWidget(QLabel("Ribasso:"), 2, 0)
                prob_layout.addWidget(QLabel(f"{prob_dist['down']*100:.1f}%"), 2, 1)
            
            self.main_layout.addLayout(prob_layout)
        
        # Aggiungi uno stretcher alla fine
        self.main_layout.addStretch()


class MarketInfoView(QWidget):
    """Widget per visualizzare informazioni di mercato."""
    
    def __init__(self, parent=None):
        """
        Inizializza il visualizzatore di informazioni.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.market_info = {}  # Informazioni di mercato
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Frame principale
        main_frame = QFrame()
        main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        main_frame.setStyleSheet(f"background-color: {style_manager.colors.CARD_BG}")
        
        self.main_layout = QVBoxLayout(main_frame)
        
        # Placeholder quando non ci sono informazioni
        self.placeholder = QLabel("Nessuna informazione disponibile")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.placeholder)
        
        # Aggiungi il frame al layout principale
        layout.addWidget(main_frame)
        
        self.setLayout(layout)
    
    def update_market_info(self, info: Optional[Dict[str, Any]]):
        """
        Aggiorna le informazioni di mercato visualizzate.
        
        Args:
            info: Dizionario con le informazioni di mercato
        """
        # Pulisci i widget esistenti
        for i in reversed(range(self.main_layout.count())): 
            self.main_layout.itemAt(i).widget().setParent(None)
        
        self.market_info = info or {}
        
        if not info:
            # Mostra placeholder se non ci sono informazioni
            self.placeholder = QLabel("Nessuna informazione disponibile")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_layout.addWidget(self.placeholder)
            return
        
        # Titolo
        if 'symbol' in info:
            symbol_obj = get_symbol(info['symbol'])
            symbol_name = symbol_obj.description if symbol_obj else info['symbol']
            
            symbol_label = QLabel(symbol_name)
            font = symbol_label.font()
            font.setBold(True)
            font.setPointSize(12)
            symbol_label.setFont(font)
            
            self.main_layout.addWidget(symbol_label)
        
        # Prezzo e variazione
        if 'price' in info:
            price_layout = QHBoxLayout()
            
            price_label = QLabel(f"{info['price']:.2f}")
            font = price_label.font()
            font.setBold(True)
            font.setPointSize(16)
            price_label.setFont(font)
            
            price_layout.addWidget(price_label)
            
            if 'change' in info:
                change = info['change']
                change_label = QLabel(f"({change:+.2f}%)")
                
                if change > 0:
                    change_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
                elif change < 0:
                    change_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
                
                price_layout.addWidget(change_label)
            
            price_layout.addStretch()
            self.main_layout.addLayout(price_layout)
        
        # Dettagli
        details_layout = QGridLayout()
        
        row = 0
        if 'high' in info and 'low' in info:
            details_layout.addWidget(QLabel("Range odierno:"), row, 0)
            details_layout.addWidget(QLabel(f"{info['low']:.2f} - {info['high']:.2f}"), row, 1)
            row += 1
        
        if 'volume' in info:
            details_layout.addWidget(QLabel("Volume:"), row, 0)
            volume = info['volume']
            
            # Formatta il volume
            if volume >= 1_000_000:
                volume_text = f"{volume/1_000_000:.2f}M"
            elif volume >= 1_000:
                volume_text = f"{volume/1_000:.2f}K"
            else:
                volume_text = str(volume)
            
            details_layout.addWidget(QLabel(volume_text), row, 1)
            row += 1
        
        # Altri dati tecnici
        if 'indicators' in info:
            indicators = info['indicators']
            
            if 'rsi' in indicators:
                details_layout.addWidget(QLabel("RSI:"), row, 0)
                rsi_value = indicators['rsi']
                rsi_label = QLabel(f"{rsi_value:.2f}")
                
                # Colora in base al valore
                if rsi_value > 70:
                    rsi_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
                elif rsi_value < 30:
                    rsi_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
                
                details_layout.addWidget(rsi_label, row, 1)
                row += 1
            
            if 'ma_trend' in indicators:
                details_layout.addWidget(QLabel("Trend MA:"), row, 0)
                ma_trend = indicators['ma_trend']
                
                if ma_trend == 'bullish':
                    ma_label = QLabel("Rialzista")
                    ma_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
                elif ma_trend == 'bearish':
                    ma_label = QLabel("Ribassista")
                    ma_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
                else:
                    ma_label = QLabel("Neutrale")
                
                details_layout.addWidget(ma_label, row, 1)
                row += 1
        
        if row > 0:
            self.main_layout.addLayout(details_layout)
        
        # Separatore
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)
        
        # Aggiungi altri dettagli se disponibili
        if 'description' in info:
            desc_label = QLabel(info['description'])
            desc_label.setWordWrap(True)
            self.main_layout.addWidget(desc_label)
        
        # Aggiungi uno stretcher alla fine
        self.main_layout.addStretch()


class PerformanceView(QWidget):
    """Widget per visualizzare le performance di trading."""
    
    def __init__(self, parent=None):
        """
        Inizializza il visualizzatore di performance.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.performance = {}  # Dati di performance
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Intestazione
        header = QLabel("Performance di Trading")
        font = header.font()
        font.setBold(True)
        font.setPointSize(12)
        header.setFont(font)
        
        layout.addWidget(header)
        
        # Frame principale
        main_frame = QFrame()
        main_frame.setFrameShape(QFrame.Shape.StyledPanel)
        main_frame.setStyleSheet(f"background-color: {style_manager.colors.CARD_BG}")
        
        self.main_layout = QVBoxLayout(main_frame)
        
        # Placeholder quando non ci sono performance
        self.placeholder = QLabel("Nessun dato di performance disponibile")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_layout.addWidget(self.placeholder)
        
        # Aggiungi il frame al layout principale
        layout.addWidget(main_frame)
        
        self.setLayout(layout)
    
    def update_performance(self, performance: Optional[Dict[str, Any]]):
        """
        Aggiorna i dati di performance visualizzati.
        
        Args:
            performance: Dizionario con i dati di performance
        """
        # Pulisci i widget esistenti
        for i in reversed(range(self.main_layout.count())): 
            self.main_layout.itemAt(i).widget().setParent(None)
        
        self.performance = performance or {}
        
        if not performance:
            # Mostra placeholder se non ci sono dati
            self.placeholder = QLabel("Nessun dato di performance disponibile")
            self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_layout.addWidget(self.placeholder)
            return
        
        # Rendimento totale
        if 'total_return' in performance:
            return_layout = QHBoxLayout()
            
            return_label = QLabel("Rendimento:")
            return_value = QLabel(f"{performance['total_return']:.2f}%")
            font = return_value.font()
            font.setBold(True)
            font.setPointSize(14)
            return_value.setFont(font)
            
            # Colora in base al valore
            if performance['total_return'] > 0:
                return_value.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['total_return'] < 0:
                return_value.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            return_layout.addWidget(return_label)
            return_layout.addWidget(return_value)
            return_layout.addStretch()
            
            self.main_layout.addLayout(return_layout)
        
        # Capitale
        if 'initial_capital' in performance and 'final_capital' in performance:
            capital_layout = QGridLayout()
            
            capital_layout.addWidget(QLabel("Capitale iniziale:"), 0, 0)
            capital_layout.addWidget(QLabel(f"${performance['initial_capital']:.2f}"), 0, 1)
            
            capital_layout.addWidget(QLabel("Capitale finale:"), 1, 0)
            final_capital_label = QLabel(f"${performance['final_capital']:.2f}")
            
            if performance['final_capital'] > performance['initial_capital']:
                final_capital_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['final_capital'] < performance['initial_capital']:
                final_capital_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            capital_layout.addWidget(final_capital_label, 1, 1)
            
            self.main_layout.addLayout(capital_layout)
        
        # Separatore
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)
        
        # Metriche chiave
        metrics_layout = QGridLayout()
        
        row = 0
        
        # Win Rate
        if 'win_rate' in performance:
            metrics_layout.addWidget(QLabel("Win Rate:"), row, 0)
            win_rate_label = QLabel(f"{performance['win_rate']:.2f}%")
            
            # Colora in base al valore
            if performance['win_rate'] >= 60:
                win_rate_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['win_rate'] >= 45:
                win_rate_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                win_rate_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            metrics_layout.addWidget(win_rate_label, row, 1)
            row += 1
        
        # Profit Factor
        if 'profit_factor' in performance:
            metrics_layout.addWidget(QLabel("Profit Factor:"), row, 0)
            pf_label = QLabel(f"{performance['profit_factor']:.2f}")
            
            # Colora in base al valore
            if performance['profit_factor'] >= 2.0:
                pf_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['profit_factor'] >= 1.3:
                pf_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                pf_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            metrics_layout.addWidget(pf_label, row, 1)
            row += 1
        
        # Sharpe Ratio
        if 'sharpe_ratio' in performance:
            metrics_layout.addWidget(QLabel("Sharpe Ratio:"), row, 0)
            sharpe_label = QLabel(f"{performance['sharpe_ratio']:.2f}")
            
            # Colora in base al valore
            if performance['sharpe_ratio'] >= 1.0:
                sharpe_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['sharpe_ratio'] >= 0.5:
                sharpe_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                sharpe_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            metrics_layout.addWidget(sharpe_label, row, 1)
            row += 1
        
        # Max Drawdown
        if 'max_drawdown' in performance:
            metrics_layout.addWidget(QLabel("Max Drawdown:"), row, 0)
            dd_label = QLabel(f"{performance['max_drawdown']:.2f}%")
            
            # Colora in base al valore (drawdown negativo è peggiore)
            if performance['max_drawdown'] <= 10:
                dd_label.setStyleSheet(f"color: {style_manager.colors.SUCCESS}")
            elif performance['max_drawdown'] <= 20:
                dd_label.setStyleSheet(f"color: {style_manager.colors.WARNING}")
            else:
                dd_label.setStyleSheet(f"color: {style_manager.colors.ERROR}")
            
            metrics_layout.addWidget(dd_label, row, 1)
            row += 1
        
        # Totale trade
        if 'total_trades' in performance:
            metrics_layout.addWidget(QLabel("Trades totali:"), row, 0)
            metrics_layout.addWidget(QLabel(str(performance['total_trades'])), row, 1)
            row += 1
        
        if 'winning_trades' in performance and 'losing_trades' in performance:
            metrics_layout.addWidget(QLabel("Trades vincenti:"), row, 0)
            metrics_layout.addWidget(QLabel(str(performance['winning_trades'])), row, 1)
            row += 1
            
            metrics_layout.addWidget(QLabel("Trades perdenti:"), row, 0)
            metrics_layout.addWidget(QLabel(str(performance['losing_trades'])), row, 1)
            row += 1
        
        if row > 0:
            self.main_layout.addLayout(metrics_layout)
        
        # Informazioni sul backtest
        if 'strategy_name' in performance:
            # Separatore
            line2 = QFrame()
            line2.setFrameShape(QFrame.Shape.HLine)
            line2.setFrameShadow(QFrame.Shadow.Sunken)
            self.main_layout.addWidget(line2)
            
            strategy_label = QLabel(f"Strategia: {performance['strategy_name']}")
            self.main_layout.addWidget(strategy_label)
            
            if 'start_date' in performance and 'end_date' in performance:
                date_range = f"Periodo: {performance['start_date']} - {performance['end_date']}"
                date_label = QLabel(date_range)
                self.main_layout.addWidget(date_label)
        
        # Aggiungi uno stretcher alla fine
        self.main_layout.addStretch()