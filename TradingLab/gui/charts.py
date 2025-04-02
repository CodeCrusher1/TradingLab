# Componenti grafici

"""
Componenti grafici per TradingLab.
Questo modulo fornisce widget per la visualizzazione di grafici finanziari e indicatori tecnici.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QGridLayout, QSizePolicy, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF, QLineF, QTimer
from PyQt6.QtGui import QColor, QPen, QBrush, QPainterPath, QPainter, QFont, QCursor

import pyqtgraph as pg
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from datetime import datetime, timedelta

from ..config import INDICATOR_PARAMS
from ..utils import app_logger
from .styles import style_manager, Theme


# Configura pyqtgraph per un look migliore
pg.setConfigOptions(antialias=True)


class TimeAxisItem(pg.AxisItem):
    """Asse X personalizzato per gestire date e orari."""
    
    def __init__(self, orientation, **kwargs):
        """
        Inizializza l'asse temporale.
        
        Args:
            orientation: Orientamento dell'asse
            **kwargs: Argomenti aggiuntivi per AxisItem
        """
        super().__init__(orientation, **kwargs)
        self.timestamps = []
    
    def set_timestamps(self, timestamps):
        """
        Imposta i timestamp da visualizzare.
        
        Args:
            timestamps: Lista di timestamp
        """
        self.timestamps = timestamps
    
    def tickStrings(self, values, scale, spacing):
        """
        Formatta i valori degli assi in stringhe di data/ora.
        
        Args:
            values: Valori asse
            scale: Scala
            spacing: Spaziatura
        
        Returns:
            Lista di stringhe formattate
        """
        if not self.timestamps or len(self.timestamps) == 0:
            return super().tickStrings(values, scale, spacing)
        
        result = []
        for v in values:
            # Converte il valore in un indice nell'array di timestamp
            idx = int(v)
            if 0 <= idx < len(self.timestamps):
                t = self.timestamps[idx]
                if isinstance(t, pd.Timestamp):
                    t = t.to_pydatetime()
                
                # Formatta in base alla spaziatura
                if spacing > 3600*24*30:  # > 30 giorni: mostra solo mese/anno
                    result.append(t.strftime('%b %Y'))
                elif spacing > 3600*24:  # > 1 giorno: mostra data
                    result.append(t.strftime('%d/%m/%Y'))
                elif spacing > 3600:  # > 1 ora: mostra data e ora
                    result.append(t.strftime('%d/%m %H:%M'))
                else:  # < 1 ora: mostra solo ora
                    result.append(t.strftime('%H:%M:%S'))
            else:
                result.append('')
        
        return result


class CandlestickItem(pg.GraphicsObject):
    """Item grafico per visualizzare candele finanziarie."""
    
    def __init__(self, data):
        """
        Inizializza l'oggetto candlestick.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        super().__init__()
        self.data = data
        self.picture = None
        self.generate_picture()
    
    def set_data(self, data):
        """
        Imposta nuovi dati per le candele.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        self.data = data
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica delle candele."""
        if self.data is None or len(self.data) == 0:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Ottieni i colori dal tema corrente
        up_color = QColor(style_manager.colors.CHART_UP)
        down_color = QColor(style_manager.colors.CHART_DOWN)
        
        w = 0.7  # Larghezza della candela
        
        for i in range(len(self.data)):
            try:
                t = i  # Indice come posizione X
                open_val = self.data['open'].iloc[i]
                high = self.data['high'].iloc[i]
                low = self.data['low'].iloc[i]
                close_val = self.data['close'].iloc[i]
                
                # Determina colore in base a candela rialzista/ribassista
                if close_val >= open_val:
                    color = up_color
                else:
                    color = down_color
                
                # Disegna corpo candela
                p.setPen(color)
                p.setBrush(color)
                
                # Calcola le coordinate rettangolo corpo candela
                rect = QRectF(t - w/2, min(open_val, close_val), 
                             w, abs(close_val - open_val))
                p.drawRect(rect)
                
                # Disegna upper shadow (stoppino superiore)
                upper_line = QLineF(t, max(open_val, close_val), t, high)
                p.drawLine(upper_line)
                
                # Disegna lower shadow (stoppino inferiore)
                lower_line = QLineF(t, min(open_val, close_val), t, low)
                p.drawLine(lower_line)
            except Exception as e:
                app_logger.error(f"Errore rendering candlestick {i}: {e}")
    
    def paint(self, p, *args):
        """
        Dipinge le candele sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene tutte le candele.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if self.data is None or len(self.data) == 0:
            return QRectF(0, 0, 1, 1)
        
        # Calcola i limiti in base ai dati
        min_x = 0
        max_x = len(self.data)
        
        # Usa valori per l'asse Y
        min_y = self.data['low'].min()
        max_y = self.data['high'].max()
        
        # Aggiungi un po' di margine
        y_margin = (max_y - min_y) * 0.05
        min_y -= y_margin
        max_y += y_margin
        
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class VolumeItem(pg.GraphicsObject):
    """Item grafico per visualizzare volumi."""
    
    def __init__(self, data):
        """
        Inizializza l'oggetto volume.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        super().__init__()
        self.data = data
        self.picture = None
        self.generate_picture()
    
    def set_data(self, data):
        """
        Imposta nuovi dati per i volumi.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        self.data = data
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica dei volumi."""
        if self.data is None or len(self.data) == 0:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Ottieni i colori dal tema corrente
        up_color = QColor(style_manager.colors.CHART_UP)
        down_color = QColor(style_manager.colors.CHART_DOWN)
        
        w = 0.7  # Larghezza della barra volume
        
        for i in range(len(self.data)):
            try:
                t = i  # Indice come posizione X
                volume = self.data['volume'].iloc[i]
                
                # Determina colore in base a candela rialzista/ribassista
                if i > 0:
                    if self.data['close'].iloc[i] >= self.data['close'].iloc[i-1]:
                        color = up_color
                    else:
                        color = down_color
                else:
                    color = up_color  # Per la prima barra
                
                # Disegna barra volume
                p.setPen(color)
                p.setBrush(color)
                
                # Calcola le coordinate del rettangolo volume
                rect = QRectF(t - w/2, 0, w, volume)
                p.drawRect(rect)
            except Exception as e:
                app_logger.error(f"Errore rendering volume {i}: {e}")
    
    def paint(self, p, *args):
        """
        Dipinge i volumi sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene tutti i volumi.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if self.data is None or len(self.data) == 0:
            return QRectF(0, 0, 1, 1)
        
        # Calcola i limiti in base ai dati
        min_x = 0
        max_x = len(self.data)
        
        # Usa valori per l'asse Y
        min_y = 0
        max_y = self.data['volume'].max() * 1.05  # Aggiungi margine
        
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class SignalItem(pg.GraphicsObject):
    """Item grafico per visualizzare segnali di trading."""
    
    def __init__(self, data, signals_column='signal'):
        """
        Inizializza l'oggetto segnali.
        
        Args:
            data: DataFrame con dati e segnali
            signals_column: Nome colonna con segnali (-1, 0, 1)
        """
        super().__init__()
        self.data = data
        self.signals_column = signals_column
        self.picture = None
        self.generate_picture()
    
    def set_data(self, data):
        """
        Imposta nuovi dati per i segnali.
        
        Args:
            data: DataFrame con dati e segnali
        """
        self.data = data
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica dei segnali."""
        if self.data is None or len(self.data) == 0 or self.signals_column not in self.data.columns:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Ottieni i colori dal tema corrente
        buy_color = QColor(style_manager.colors.SUCCESS)
        sell_color = QColor(style_manager.colors.ERROR)
        
        # Definisci le dimensioni dei marker
        marker_size = 15
        
        for i in range(len(self.data)):
            try:
                signal = self.data[self.signals_column].iloc[i]
                if signal == 0:
                    continue  # Nessun segnale
                
                t = i  # Indice come posizione X
                price = self.data['close'].iloc[i]
                
                if signal == 1:  # Segnale di acquisto
                    p.setPen(QPen(buy_color, 2))
                    # Disegna triangolo verso l'alto
                    path = QPainterPath()
                    path.moveTo(t, price - marker_size)
                    path.lineTo(t - marker_size/2, price - marker_size/2)
                    path.lineTo(t + marker_size/2, price - marker_size/2)
                    path.closeSubpath()
                    p.drawPath(path)
                
                elif signal == -1:  # Segnale di vendita
                    p.setPen(QPen(sell_color, 2))
                    # Disegna triangolo verso il basso
                    path = QPainterPath()
                    path.moveTo(t, price + marker_size)
                    path.lineTo(t - marker_size/2, price + marker_size/2)
                    path.lineTo(t + marker_size/2, price + marker_size/2)
                    path.closeSubpath()
                    p.drawPath(path)
                
            except Exception as e:
                app_logger.error(f"Errore rendering segnale {i}: {e}")
    
    def paint(self, p, *args):
        """
        Dipinge i segnali sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene tutti i segnali.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if self.data is None or len(self.data) == 0:
            return QRectF(0, 0, 1, 1)
        
        # Calcola i limiti in base ai dati price
        min_x = 0
        max_x = len(self.data)
        
        # Usa valori per l'asse Y
        min_y = self.data['low'].min()
        max_y = self.data['high'].max()
        
        # Aggiungi un po' di margine
        y_margin = (max_y - min_y) * 0.1
        min_y -= y_margin
        max_y += y_margin
        
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class PredictionItem(pg.GraphicsObject):
    """Item grafico per visualizzare previsioni."""
    
    def __init__(self, prediction=None):
        """
        Inizializza l'oggetto previsioni.
        
        Args:
            prediction: Dizionario con dati della previsione
        """
        super().__init__()
        self.prediction = prediction
        self.last_candle_idx = 0
        self.last_price = 0
        self.picture = None
        self.generate_picture()
    
    def set_prediction(self, prediction, last_candle_idx, last_price):
        """
        Imposta nuovi dati per la previsione.
        
        Args:
            prediction: Dizionario con dati della previsione
            last_candle_idx: Indice dell'ultima candela
            last_price: Prezzo dell'ultima candela
        """
        self.prediction = prediction
        self.last_candle_idx = last_candle_idx
        self.last_price = last_price
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica della previsione."""
        if self.prediction is None:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Ottieni i colori dal tema corrente
        up_color = QColor(style_manager.colors.SUCCESS)
        down_color = QColor(style_manager.colors.ERROR)
        neutral_color = QColor(style_manager.colors.NEUTRAL)
        
        try:
            # Posizione X per la previsione (dopo l'ultima candela)
            x_pos = self.last_candle_idx
            
            # Ottieni dati dalla previsione
            direction = self.prediction.get('direction', 0)
            entry_price = self.prediction.get('entry_price', self.last_price)
            tp_price = self.prediction.get('tp_price')
            sl_price = self.prediction.get('sl_price')
            
            # Determina colore in base alla direzione
            if direction > 0:
                color = up_color
            elif direction < 0:
                color = down_color
            else:
                color = neutral_color
            
            # Disegna simbolo di direzione
            p.setPen(QPen(color, 2))
            p.setBrush(QBrush(color))
            
            # Disegna un cerchio per indicare la posizione della previsione
            circle_radius = 8
            p.drawEllipse(QPointF(x_pos, entry_price), circle_radius, circle_radius)
            
            # Disegna linee per TP e SL
            if tp_price is not None and sl_price is not None:
                dash_pen = QPen(color, 1, Qt.PenStyle.DashLine)
                p.setPen(dash_pen)
                
                # Linea orizzontale per TP
                p.drawLine(QLineF(x_pos - circle_radius, tp_price, 
                                 x_pos + circle_radius * 3, tp_price))
                
                # Linea orizzontale per SL
                p.drawLine(QLineF(x_pos - circle_radius, sl_price, 
                                 x_pos + circle_radius * 3, sl_price))
                
                # Etichette per TP e SL
                font = p.font()
                font.setPointSize(8)
                p.setFont(font)
                
                p.drawText(QPointF(x_pos + circle_radius * 3 + 5, tp_price), "TP")
                p.drawText(QPointF(x_pos + circle_radius * 3 + 5, sl_price), "SL")
            
        except Exception as e:
            app_logger.error(f"Errore rendering previsione: {e}")
    
    def paint(self, p, *args):
        """
        Dipinge la previsione sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene la previsione.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if self.prediction is None:
            return QRectF(0, 0, 1, 1)
        
        # Calcola i limiti in base ai dati
        x_pos = self.last_candle_idx
        
        entry_price = self.prediction.get('entry_price', self.last_price)
        tp_price = self.prediction.get('tp_price', entry_price * 1.05)
        sl_price = self.prediction.get('sl_price', entry_price * 0.95)
        
        min_y = min(entry_price, tp_price, sl_price) * 0.98
        max_y = max(entry_price, tp_price, sl_price) * 1.02
        
        return QRectF(x_pos - 20, min_y, 100, max_y - min_y)


class SupportResistanceItem(pg.GraphicsObject):
    """Item grafico per visualizzare livelli di supporto e resistenza."""
    
    def __init__(self, levels=None):
        """
        Inizializza l'oggetto supporti/resistenze.
        
        Args:
            levels: Lista di dizionari con livelli
        """
        super().__init__()
        self.levels = levels or []
        self.data_length = 0
        self.picture = None
        self.generate_picture()
    
    def set_levels(self, levels, data_length):
        """
        Imposta nuovi livelli di supporto/resistenza.
        
        Args:
            levels: Lista di dizionari con livelli
            data_length: Lunghezza dei dati (numero di candele)
        """
        self.levels = levels
        self.data_length = data_length
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica dei livelli."""
        if not self.levels or self.data_length == 0:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Ottieni i colori dal tema corrente
        support_color = QColor(style_manager.colors.SUCCESS)
        resistance_color = QColor(style_manager.colors.ERROR)
        
        for level in self.levels:
            try:
                price = level.get('price', 0)
                level_type = level.get('type', 'support')
                
                # Scegli colore in base al tipo di livello
                if level_type == 'support':
                    color = support_color
                else:  # resistance
                    color = resistance_color
                
                # Imposta penna tratteggiata
                pen = QPen(color, 1, Qt.PenStyle.DashLine)
                p.setPen(pen)
                
                # Disegna linea orizzontale per tutto il grafico
                p.drawLine(QLineF(0, price, self.data_length, price))
                
                # Aggiungi etichetta
                font = p.font()
                font.setPointSize(8)
                p.setFont(font)
                
                # Posiziona etichetta alla fine della linea
                label_text = f"{level_type.capitalize()}: {price:.2f}"
                p.drawText(QPointF(self.data_length + 5, price), label_text)
                
            except Exception as e:
                app_logger.error(f"Errore rendering livello: {e}")
    
    def paint(self, p, *args):
        """
        Dipinge i livelli sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene tutti i livelli.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if not self.levels or self.data_length == 0:
            return QRectF(0, 0, 1, 1)
        
        # Calcola i limiti in base ai dati
        min_x = 0
        max_x = self.data_length + 100  # Aggiungi spazio per le etichette
        
        # Cerca min/max prezzi
        prices = [level.get('price', 0) for level in self.levels]
        min_y = min(prices) * 0.98
        max_y = max(prices) * 1.02
        
        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class InfoTextItem(pg.GraphicsObject):
    """Item grafico per visualizzare informazioni sul grafico."""
    
    def __init__(self, text="", anchor=(0, 0)):
        """
        Inizializza l'oggetto testo.
        
        Args:
            text: Testo da visualizzare
            anchor: Posizione di ancoraggio (x, y)
        """
        super().__init__()
        self.text = text
        self.anchor = anchor
        self.picture = None
        self.generate_picture()
    
    def set_text(self, text):
        """
        Imposta nuovo testo.
        
        Args:
            text: Testo da visualizzare
        """
        self.text = text
        self.generate_picture()
        self.update()
    
    def generate_picture(self):
        """Genera la rappresentazione grafica del testo."""
        if not self.text:
            return
        
        # Crea un nuovo QPicture
        self.picture = QPainter()
        p = QPainter()
        
        # Imposta font e colore
        font = QFont()
        font.setPointSize(10)
        p.setFont(font)
        
        text_color = QColor(style_manager.colors.TEXT_PRIMARY)
        p.setPen(text_color)
        
        # Posizione del testo
        p.drawText(QPointF(self.anchor[0], self.anchor[1]), self.text)
    
    def paint(self, p, *args):
        """
        Dipinge il testo sul widget.
        
        Args:
            p: Painter
            *args: Argomenti aggiuntivi
        """
        if self.picture is not None:
            self.picture.begin(p)
            self.picture.end()
    
    def boundingRect(self):
        """
        Definisce il rettangolo che contiene il testo.
        
        Returns:
            QRectF con i limiti dell'oggetto
        """
        if not self.text:
            return QRectF(0, 0, 1, 1)
        
        # Stima della dimensione del testo
        text_width = len(self.text) * 7  # Stima grossolana
        return QRectF(self.anchor[0], self.anchor[1] - 15, text_width, 20)


class CandlestickChart(QWidget):
    """Widget per visualizzare grafici a candele con indicatori."""
    
    # Segnale emesso quando si clicca su un punto nel grafico
    pointClicked = pyqtSignal(int, float)
    
    def __init__(self, parent=None):
        """
        Inizializza il grafico a candele.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.data = None
        self.indicators = {}
        self.prediction = None
        self.support_resistance_levels = []
        self.cursor_enabled = True
        
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Crea l'area grafica principale
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground(style_manager.colors.CARD_BG)
        
        # Configura l'asse X per visualizzare date/orari
        self.time_axis = TimeAxisItem(orientation='bottom')
        self.graphWidget.setAxisItems({'bottom': self.time_axis})
        
        # Crea gli elementi grafici
        self.candlestick_item = CandlestickItem(None)
        self.signal_item = SignalItem(None)
        self.prediction_item = PredictionItem()
        self.support_resistance_item = SupportResistanceItem()
        self.cursor_text = InfoTextItem()
        
        # Imposta il cursore per mostrare informazioni
        self.cursor_line_v = pg.InfiniteLine(angle=90, movable=False)
        self.cursor_line_h = pg.InfiniteLine(angle=0, movable=False)
        
        # Nascondi il cursore inizialmente
        self.cursor_line_v.hide()
        self.cursor_line_h.hide()
        
        # Aggiungi gli elementi al grafico
        self.graphWidget.addItem(self.candlestick_item)
        self.graphWidget.addItem(self.signal_item)
        self.graphWidget.addItem(self.prediction_item)
        self.graphWidget.addItem(self.support_resistance_item)
        self.graphWidget.addItem(self.cursor_text)
        self.graphWidget.addItem(self.cursor_line_v)
        self.graphWidget.addItem(self.cursor_line_h)
        
        # Configura spazio per indicatori aggiuntivi
        self.indicators_layout = QVBoxLayout()
        
        # Aggiungi i widget al layout
        layout.addWidget(self.graphWidget, stretch=7)
        layout.addLayout(self.indicators_layout, stretch=3)
        
        # Dimensione iniziale consigliata
        self.setMinimumSize(600, 400)
        
        # Connetti i segnali
        self.graphWidget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self.graphWidget.scene().sigMouseClicked.connect(self._on_mouse_clicked)
    
    def set_data(self, data: pd.DataFrame):
        """
        Imposta i dati del grafico.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        self.data = data
        
        if data is None or data.empty:
            return
        
        # Aggiorna i timestamp per l'asse X
        if 'timestamp' in data.columns:
            self.time_axis.set_timestamps(data['timestamp'].tolist())
        
        # Aggiorna gli elementi grafici
        self.candlestick_item.set_data(data)
        self.signal_item.set_data(data)
        
        # Aggiorna il prediction_item se c'è una previsione
        if self.prediction:
            self.prediction_item.set_prediction(
                self.prediction, 
                len(data) - 1,  # Indice ultima candela
                data['close'].iloc[-1]  # Prezzo ultima candela
            )
        
        # Aggiorna i livelli di supporto/resistenza
        self.support_resistance_item.set_levels(self.support_resistance_levels, len(data))
        
        # Aggiorna le visualizzazioni degli indicatori
        self._update_indicators()
        
        # Imposta i limiti del grafico
        self._update_view_limits()
    
    def set_prediction(self, prediction: Dict[str, Any]):
        """
        Imposta i dati di previsione.
        
        Args:
            prediction: Dizionario con dati previsione
        """
        self.prediction = prediction
        
        # Aggiorna solo se ci sono dati
        if self.data is not None and not self.data.empty:
            self.prediction_item.set_prediction(
                prediction, 
                len(self.data) - 1,  # Indice ultima candela
                self.data['close'].iloc[-1]  # Prezzo ultima candela
            )
    
    def set_support_resistance_levels(self, levels: List[Dict[str, Any]]):
        """
        Imposta livelli di supporto e resistenza.
        
        Args:
            levels: Lista di livelli
        """
        self.support_resistance_levels = levels
        
        # Aggiorna solo se ci sono dati
        if self.data is not None and not self.data.empty:
            self.support_resistance_item.set_levels(levels, len(self.data))
    
    def enable_cursor(self, enabled: bool = True):
        """
        Abilita/disabilita il cursore interattivo.
        
        Args:
            enabled: Se il cursore deve essere abilitato
        """
        self.cursor_enabled = enabled
        
        if not enabled:
            self.cursor_line_v.hide()
            self.cursor_line_h.hide()
            self.cursor_text.set_text("")
    
    def _update_view_limits(self):
        """Aggiorna i limiti di visualizzazione del grafico."""
        if self.data is None or self.data.empty:
            return
        
        # Imposta i limiti dell'asse X
        self.graphWidget.setXRange(0, len(self.data) - 1)
        
        # Imposta i limiti dell'asse Y basati su high/low
        min_price = self.data['low'].min()
        max_price = self.data['high'].max()
        price_range = max_price - min_price
        
        # Aggiungi un margine del 5%
        margin = price_range * 0.05
        self.graphWidget.setYRange(min_price - margin, max_price + margin)
    
    def add_indicator(self, name: str, data: pd.DataFrame, columns: List[str], colors: Optional[List[str]] = None):
        """
        Aggiunge un indicatore al grafico.
        
        Args:
            name: Nome dell'indicatore
            data: DataFrame con dati dell'indicatore
            columns: Colonne da visualizzare
            colors: Colori per le linee (opzionale)
        """
        if not columns:
            return
        
        # Se non ci sono colori specificati, usa quelli del tema
        if colors is None:
            colors = [
                style_manager.colors.INDICATOR_LINE_1,
                style_manager.colors.INDICATOR_LINE_2,
                style_manager.colors.INDICATOR_LINE_3,
                style_manager.colors.INDICATOR_LINE_4,
                style_manager.colors.INDICATOR_LINE_5
            ]
        
        # Memorizza i dettagli dell'indicatore
        self.indicators[name] = {
            'data': data,
            'columns': columns,
            'colors': colors,
            'plot_item': None,
            'overlay': False  # Default: indicatore separato
        }
        
        # Aggiorna gli indicatori visualizzati
        self._update_indicators()
    
    def add_overlay_indicator(self, name: str, data: pd.DataFrame, columns: List[str], colors: Optional[List[str]] = None):
        """
        Aggiunge un indicatore sovrapposto al grafico principale.
        
        Args:
            name: Nome dell'indicatore
            data: DataFrame con dati dell'indicatore
            columns: Colonne da visualizzare
            colors: Colori per le linee (opzionale)
        """
        if not columns:
            return
        
        # Se non ci sono colori specificati, usa quelli del tema
        if colors is None:
            colors = [
                style_manager.colors.INDICATOR_LINE_1,
                style_manager.colors.INDICATOR_LINE_2,
                style_manager.colors.INDICATOR_LINE_3,
                style_manager.colors.INDICATOR_LINE_4,
                style_manager.colors.INDICATOR_LINE_5
            ]
        
        # Memorizza i dettagli dell'indicatore
        self.indicators[name] = {
            'data': data,
            'columns': columns,
            'colors': colors,
            'plot_item': None,
            'overlay': True  # Questo è un overlay
        }
        
        # Aggiorna gli indicatori visualizzati
        self._update_indicators()
    
    def remove_indicator(self, name: str):
        """
        Rimuove un indicatore dal grafico.
        
        Args:
            name: Nome dell'indicatore da rimuovere
        """
        if name in self.indicators:
            indicator = self.indicators[name]
            
            # Rimuovi l'elemento dal grafico se esiste
            if indicator['plot_item'] is not None:
                if indicator['overlay']:
                    # Rimuovi le curve dal grafico principale
                    for curve in indicator['plot_item']:
                        self.graphWidget.removeItem(curve)
                else:
                    # Rimuovi il widget dal layout
                    indicator['plot_item'].setParent(None)
                    
                    # Rimuovi il widget dal layout
                    for i in range(self.indicators_layout.count()):
                        item = self.indicators_layout.itemAt(i)
                        if item.widget() == indicator['plot_item']:
                            self.indicators_layout.removeItem(item)
                            break
            
            # Rimuovi dall'elenco
            del self.indicators[name]
    
    def _update_indicators(self):
        """Aggiorna tutti gli indicatori visualizzati."""
        # Prima rimuovi tutti gli indicatori esistenti
        for name, indicator in self.indicators.items():
            if indicator['plot_item'] is not None:
                if indicator['overlay']:
                    # Rimuovi le curve dal grafico principale
                    for curve in indicator['plot_item']:
                        self.graphWidget.removeItem(curve)
                else:
                    # Rimuovi il widget dal layout
                    indicator['plot_item'].setParent(None)
            
            indicator['plot_item'] = None
        
        # Pulisci il layout degli indicatori
        for i in reversed(range(self.indicators_layout.count())):
            item = self.indicators_layout.itemAt(i)
            if item.widget():
                item.widget().setParent(None)
        
        # Ora aggiungi gli indicatori aggiornati
        for name, indicator in self.indicators.items():
            data = indicator['data']
            columns = indicator['columns']
            colors = indicator['colors']
            overlay = indicator['overlay']
            
            if data is None or data.empty:
                continue
            
            if overlay:
                # Sovrapponi l'indicatore al grafico principale
                curves = []
                for i, col in enumerate(columns):
                    if col in data.columns:
                        color = colors[i % len(colors)]
                        curve = pg.PlotDataItem(
                            x=list(range(len(data))),
                            y=data[col],
                            pen=pg.mkPen(color=color, width=1.5)
                        )
                        self.graphWidget.addItem(curve)
                        curves.append(curve)
                
                indicator['plot_item'] = curves
            
            else:
                # Crea un grafico separato per l'indicatore
                indicator_plot = pg.PlotWidget()
                indicator_plot.setBackground(style_manager.colors.CARD_BG)
                indicator_plot.setFixedHeight(120)
                
                # Usa lo stesso asse X del grafico principale
                indicator_axis = TimeAxisItem(orientation='bottom')
                indicator_axis.set_timestamps(self.time_axis.timestamps)
                indicator_plot.setAxisItems({'bottom': indicator_axis})
                
                # Aggiungi il titolo
                title_label = QLabel(name)
                title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                title_label.setStyleSheet(f"color: {style_manager.colors.TEXT_PRIMARY}")
                
                # Crea un widget container per l'indicatore
                container = QWidget()
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(0, 0, 0, 0)
                container_layout.addWidget(title_label)
                container_layout.addWidget(indicator_plot)
                
                # Aggiungi le curve
                for i, col in enumerate(columns):
                    if col in data.columns:
                        color = colors[i % len(colors)]
                        indicator_plot.plot(
                            x=list(range(len(data))),
                            y=data[col],
                            pen=pg.mkPen(color=color, width=1.5),
                            name=col
                        )
                
                # Imposta i limiti dell'asse X per corrispondere al grafico principale
                indicator_plot.setXRange(0, len(data) - 1)
                
                # Collegati all'evento di zoom del grafico principale
                # (in una implementazione completa, dovresti sincronizzare lo zoom)
                
                # Aggiungi al layout
                self.indicators_layout.addWidget(container)
                
                indicator['plot_item'] = container
    
    def clear_all_indicators(self):
        """Rimuove tutti gli indicatori dal grafico."""
        indicator_names = list(self.indicators.keys())
        for name in indicator_names:
            self.remove_indicator(name)
    
    def zoom_to_last_n_candles(self, n: int = 100):
        """
        Esegue lo zoom alle ultime N candele.
        
        Args:
            n: Numero di candele da visualizzare
        """
        if self.data is None or len(self.data) == 0:
            return
        
        data_len = len(self.data)
        if n >= data_len:
            self._update_view_limits()  # Mostra tutto
            return
        
        # Calcola l'intervallo da visualizzare
        start_idx = max(0, data_len - n)
        end_idx = data_len - 1
        
        # Imposta i limiti dell'asse X
        self.graphWidget.setXRange(start_idx, end_idx)
        
        # Imposta i limiti dell'asse Y basati sui dati visibili
        visible_data = self.data.iloc[start_idx:end_idx+1]
        min_price = visible_data['low'].min()
        max_price = visible_data['high'].max()
        price_range = max_price - min_price
        
        # Aggiungi un margine del 5%
        margin = price_range * 0.05
        self.graphWidget.setYRange(min_price - margin, max_price + margin)
    
    def _on_mouse_moved(self, pos):
        """
        Gestisce il movimento del mouse sul grafico.
        
        Args:
            pos: Posizione del mouse
        """
        if not self.cursor_enabled or self.data is None or self.data.empty:
            return
        
        # Converti le coordinate del mouse in coordinate del grafico
        mouse_point = self.graphWidget.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Arrotonda la coordinata x al valore intero più vicino (indice candela)
        x_idx = int(round(x))
        
        # Verifica se l'indice è valido
        if 0 <= x_idx < len(self.data):
            # Aggiorna posizione delle linee del cursore
            self.cursor_line_v.setPos(x_idx)
            self.cursor_line_h.setPos(y)
            self.cursor_line_v.show()
            self.cursor_line_h.show()
            
            # Formato data/ora
            timestamp = self.data['timestamp'].iloc[x_idx]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
            
            date_str = timestamp.strftime('%d/%m/%Y')
            time_str = timestamp.strftime('%H:%M:%S')
            
            # Ottieni dati OHLCV della candela
            candle_data = self.data.iloc[x_idx]
            
            # Crea testo informativo
            info_text = (f"Data: {date_str} {time_str}\n"
                        f"O: {candle_data['open']:.2f} H: {candle_data['high']:.2f} "
                        f"L: {candle_data['low']:.2f} C: {candle_data['close']:.2f} "
                        f"V: {int(candle_data['volume'])}")
            
            # Aggiorna il testo del cursore
            self.cursor_text.set_text(info_text)
            
            # Posiziona il testo in alto a destra (arbitrario)
            self.cursor_text.anchor = (5, 20)
        else:
            # Nascondi il cursore se fuori dal range
            self.cursor_line_v.hide()
            self.cursor_line_h.hide()
            self.cursor_text.set_text("")
    
    def _on_mouse_clicked(self, event):
        """
        Gestisce il click del mouse sul grafico.
        
        Args:
            event: Evento click
        """
        if self.data is None or self.data.empty:
            return
        
        # Ottieni il punto cliccato nelle coordinate del grafico
        pos = event.scenePos()
        mouse_point = self.graphWidget.getPlotItem().vb.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Arrotonda la coordinata x al valore intero più vicino (indice candela)
        x_idx = int(round(x))
        
        # Verifica se l'indice è valido
        if 0 <= x_idx < len(self.data):
            # Emetti il segnale con l'indice e il prezzo
            self.pointClicked.emit(x_idx, y)


class VolumeChart(QWidget):
    """Widget per visualizzare grafici di volume."""
    
    def __init__(self, parent=None):
        """
        Inizializza il grafico volume.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.data = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Crea l'area grafica
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground(style_manager.colors.CARD_BG)
        self.graphWidget.setMaximumHeight(120)
        
        # Configura l'asse X per visualizzare date/orari
        self.time_axis = TimeAxisItem(orientation='bottom')
        self.graphWidget.setAxisItems({'bottom': self.time_axis})
        
        # Crea l'elemento volume
        self.volume_item = VolumeItem(None)
        
        # Aggiungi l'elemento al grafico
        self.graphWidget.addItem(self.volume_item)
        
        # Titolo
        title_label = QLabel("Volume")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"color: {style_manager.colors.TEXT_PRIMARY}")
        
        # Aggiungi i widget al layout
        layout.addWidget(title_label)
        layout.addWidget(self.graphWidget)
        
        self.setLayout(layout)
    
    def set_data(self, data: pd.DataFrame):
        """
        Imposta i dati del grafico volume.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        self.data = data
        
        if data is None or data.empty:
            return
        
        # Aggiorna i timestamp per l'asse X
        if 'timestamp' in data.columns:
            self.time_axis.set_timestamps(data['timestamp'].tolist())
        
        # Aggiorna l'elemento volume
        self.volume_item.set_data(data)
        
        # Imposta i limiti del grafico
        self._update_view_limits()
    
    def _update_view_limits(self):
        """Aggiorna i limiti di visualizzazione del grafico."""
        if self.data is None or self.data.empty:
            return
        
        # Imposta i limiti dell'asse X
        self.graphWidget.setXRange(0, len(self.data) - 1)
        
        # Imposta i limiti dell'asse Y basati sul volume
        max_volume = self.data['volume'].max()
        
        # Aggiungi un margine del 10%
        margin = max_volume * 0.1
        self.graphWidget.setYRange(0, max_volume + margin)
    
    def sync_with_chart(self, chart: CandlestickChart):
        """
        Sincronizza questo grafico con un grafico candlestick.
        
        Args:
            chart: Grafico candlestick da sincronizzare
        """
        # Ottieni la regione visibile dal grafico principale
        region = chart.graphWidget.getViewBox().viewRange()
        
        # Imposta la stessa regione X per questo grafico
        self.graphWidget.setXRange(region[0][0], region[0][1])


class ChartControlPanel(QWidget):
    """Pannello di controllo per i grafici."""
    
    # Segnali
    timeRangeChanged = pyqtSignal(int)
    indicatorToggled = pyqtSignal(str, bool)
    chartTypeChanged = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Inizializza il pannello di controllo.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QHBoxLayout(self)
        
        # Tipo di grafico
        chart_type_layout = QVBoxLayout()
        chart_type_layout.addWidget(QLabel("Tipo di grafico:"))
        
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItem("Candlestick")
        self.chart_type_combo.addItem("OHLC")
        self.chart_type_combo.addItem("Line")
        
        chart_type_layout.addWidget(self.chart_type_combo)
        
        # Intervallo di visualizzazione
        time_range_layout = QVBoxLayout()
        time_range_layout.addWidget(QLabel("Visualizza:"))
        
        self.time_range_combo = QComboBox()
        self.time_range_combo.addItem("Tutte le candele", -1)
        self.time_range_combo.addItem("Ultime 50 candele", 50)
        self.time_range_combo.addItem("Ultime 100 candele", 100)
        self.time_range_combo.addItem("Ultime 200 candele", 200)
        
        time_range_layout.addWidget(self.time_range_combo)
        
        # Indicatori rapidi
        indicators_layout = QVBoxLayout()
        indicators_layout.addWidget(QLabel("Indicatori:"))
        
        indicators_controls = QHBoxLayout()
        
        self.ema_check = QCheckBox("EMA")
        self.ema_check.setChecked(True)
        
        self.volume_check = QCheckBox("Volume")
        self.volume_check.setChecked(True)
        
        self.bb_check = QCheckBox("Bollinger")
        
        indicators_controls.addWidget(self.ema_check)
        indicators_controls.addWidget(self.volume_check)
        indicators_controls.addWidget(self.bb_check)
        
        indicators_layout.addLayout(indicators_controls)
        
        # Aggiungi i layout al layout principale
        layout.addLayout(chart_type_layout)
        layout.addLayout(time_range_layout)
        layout.addLayout(indicators_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
        # Connetti i segnali
        self.chart_type_combo.currentTextChanged.connect(
            lambda text: self.chartTypeChanged.emit(text.lower())
        )
        
        self.time_range_combo.currentIndexChanged.connect(
            lambda idx: self.timeRangeChanged.emit(self.time_range_combo.itemData(idx))
        )
        
        self.ema_check.stateChanged.connect(
            lambda state: self.indicatorToggled.emit("ema", state == Qt.CheckState.Checked)
        )
        
        self.volume_check.stateChanged.connect(
            lambda state: self.indicatorToggled.emit("volume", state == Qt.CheckState.Checked)
        )
        
        self.bb_check.stateChanged.connect(
            lambda state: self.indicatorToggled.emit("bollinger", state == Qt.CheckState.Checked)
        )


class ChartWidget(QWidget):
    """Widget completo per visualizzazione di grafici finanziari."""
    
    def __init__(self, parent=None):
        """
        Inizializza il widget per grafici finanziari.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.data = None
        self.volume_visible = True
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Controlli del grafico
        self.control_panel = ChartControlPanel()
        
        # Grafico candlestick
        self.candlestick_chart = CandlestickChart()
        
        # Grafico volume
        self.volume_chart = VolumeChart()
        
        # Aggiungi i widget al layout
        layout.addWidget(self.control_panel)
        layout.addWidget(self.candlestick_chart, stretch=8)
        layout.addWidget(self.volume_chart, stretch=2)
        
        self.setLayout(layout)
        
        # Connetti i segnali
        self.control_panel.timeRangeChanged.connect(self._on_time_range_changed)
        self.control_panel.indicatorToggled.connect(self._on_indicator_toggled)
        self.control_panel.chartTypeChanged.connect(self._on_chart_type_changed)
    
    def set_data(self, data: pd.DataFrame):
        """
        Imposta i dati del grafico.
        
        Args:
            data: DataFrame con dati OHLCV
        """
        self.data = data
        
        if data is None or data.empty:
            return
        
        # Aggiorna i grafici
        self.candlestick_chart.set_data(data)
        
        if self.volume_visible:
            self.volume_chart.set_data(data)
        
        # Sincronizza il grafico volume con il grafico candlestick
        self.volume_chart.sync_with_chart(self.candlestick_chart)
        
        # Aggiungi indicatori predefiniti
        self._add_default_indicators()
    
    def set_prediction(self, prediction: Dict[str, Any]):
        """
        Imposta i dati di previsione.
        
        Args:
            prediction: Dizionario con dati previsione
        """
        self.candlestick_chart.set_prediction(prediction)
    
    def set_support_resistance_levels(self, levels: List[Dict[str, Any]]):
        """
        Imposta livelli di supporto e resistenza.
        
        Args:
            levels: Lista di livelli
        """
        self.candlestick_chart.set_support_resistance_levels(levels)
    
    def _add_default_indicators(self):
        """Aggiunge indicatori predefiniti al grafico."""
        if self.data is None or self.data.empty:
            return
        
        # Calcola EMA
        if 'ema_21' not in self.data.columns:
            self.data['ema_21'] = self.data['close'].ewm(span=21, adjust=False).mean()
        
        if 'ema_50' not in self.data.columns:
            self.data['ema_50'] = self.data['close'].ewm(span=50, adjust=False).mean()
        
        if 'ema_200' not in self.data.columns:
            self.data['ema_200'] = self.data['close'].ewm(span=200, adjust=False).mean()
        
        # Aggiungi EMA come overlay al grafico
        ema_columns = ['ema_21', 'ema_50', 'ema_200']
        ema_colors = [
            style_manager.colors.INDICATOR_LINE_1,
            style_manager.colors.INDICATOR_LINE_2,
            style_manager.colors.INDICATOR_LINE_3
        ]
        
        self.candlestick_chart.add_overlay_indicator('EMA', self.data, ema_columns, ema_colors)
    
    def _on_time_range_changed(self, num_candles: int):
        """
        Gestisce il cambio di intervallo temporale.
        
        Args:
            num_candles: Numero di candele da visualizzare
        """
        if num_candles < 0:
            # Mostra tutte le candele
            self.candlestick_chart._update_view_limits()
        else:
            # Mostra le ultime N candele
            self.candlestick_chart.zoom_to_last_n_candles(num_candles)
        
        # Sincronizza il grafico volume
        self.volume_chart.sync_with_chart(self.candlestick_chart)
    
    def _on_indicator_toggled(self, indicator: str, visible: bool):
        """
        Gestisce l'attivazione/disattivazione di un indicatore.
        
        Args:
            indicator: Nome dell'indicatore
            visible: Se l'indicatore deve essere visibile
        """
        if indicator == "volume":
            self.volume_visible = visible
            self.volume_chart.setVisible(visible)
        
        elif indicator == "ema":
            if visible:
                # Aggiungi EMA al grafico se non è già presente
                if 'EMA' not in self.candlestick_chart.indicators:
                    ema_columns = ['ema_21', 'ema_50', 'ema_200']
                    ema_colors = [
                        style_manager.colors.INDICATOR_LINE_1,
                        style_manager.colors.INDICATOR_LINE_2,
                        style_manager.colors.INDICATOR_LINE_3
                    ]
                    
                    self.candlestick_chart.add_overlay_indicator('EMA', self.data, ema_columns, ema_colors)
            else:
                # Rimuovi EMA dal grafico
                self.candlestick_chart.remove_indicator('EMA')
        
        elif indicator == "bollinger":
            if visible:
                # Calcola e aggiungi Bollinger Bands se non già presenti
                if 'bb_middle_20' not in self.data.columns:
                    # Calcola SMA per BB
                    self.data['bb_middle_20'] = self.data['close'].rolling(window=20).mean()
                    std = self.data['close'].rolling(window=20).std()
                    
                    # Calcola le bande
                    self.data['bb_upper_20'] = self.data['bb_middle_20'] + (2 * std)
                    self.data['bb_lower_20'] = self.data['bb_middle_20'] - (2 * std)
                
                # Aggiungi BB al grafico
                bb_columns = ['bb_middle_20', 'bb_upper_20', 'bb_lower_20']
                bb_colors = [
                    style_manager.colors.NEUTRAL,
                    style_manager.colors.WARNING,
                    style_manager.colors.WARNING
                ]
                
                self.candlestick_chart.add_overlay_indicator('Bollinger', self.data, bb_columns, bb_colors)
            else:
                # Rimuovi BB dal grafico
                self.candlestick_chart.remove_indicator('Bollinger')
    
    def _on_chart_type_changed(self, chart_type: str):
        """
        Gestisce il cambio di tipo di grafico.
        
        Args:
            chart_type: Tipo di grafico
        """
        # Nota: l'implementazione completa richiederebbe diversi tipi di grafici
        # Per semplicità, in questa versione supportiamo solo candlestick
        app_logger.info(f"Cambio tipo grafico a {chart_type} (non implementato)")


class BacktestResultChart(QWidget):
    """Widget per visualizzare i risultati di backtest."""
    
    def __init__(self, parent=None):
        """
        Inizializza il grafico risultati backtest.
        
        Args:
            parent: Widget genitore
        """
        super().__init__(parent)
        self.backtest_data = None
        self.initUI()
    
    def initUI(self):
        """Inizializza l'interfaccia del widget."""
        layout = QVBoxLayout(self)
        
        # Grafico equity
        equity_label = QLabel("Curva Equity")
        equity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.equity_graph = pg.PlotWidget()
        self.equity_graph.setBackground(style_manager.colors.CARD_BG)
        
        # Grafico drawdown
        drawdown_label = QLabel("Drawdown")
        drawdown_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.drawdown_graph = pg.PlotWidget()
        self.drawdown_graph.setBackground(style_manager.colors.CARD_BG)
        self.drawdown_graph.setMaximumHeight(120)
        
        # Aggiungi i widget al layout
        layout.addWidget(equity_label)
        layout.addWidget(self.equity_graph, stretch=7)
        layout.addWidget(drawdown_label)
        layout.addWidget(self.drawdown_graph, stretch=3)
        
        self.setLayout(layout)
    
    def set_backtest_results(self, results: Dict[str, Any]):
        """
        Imposta i risultati del backtest.
        
        Args:
            results: Dizionario con risultati backtest
        """
        self.backtest_data = results
        
        if not results or 'equity_curve' not in results or 'drawdown_curve' not in results:
            return
        
        # Pulisci i grafici
        self.equity_graph.clear()
        self.drawdown_graph.clear()
        
        # Estrai i dati
        equity_curve = results['equity_curve']
        drawdown_curve = results['drawdown_curve']
        
        if not equity_curve or not drawdown_curve:
            return
        
        # Prepara i dati per il grafico equity
        dates = [item[0] for item in equity_curve]
        equity_values = [item[1] for item in equity_curve]
        
        # Crea l'asse X personalizzato per le date
        time_axis_equity = TimeAxisItem(orientation='bottom')
        time_axis_equity.set_timestamps(dates)
        self.equity_graph.setAxisItems({'bottom': time_axis_equity})
        
        # Grafico equity
        equity_curve_item = pg.PlotDataItem(
            x=list(range(len(equity_values))),
            y=equity_values,
            pen=pg.mkPen(color=style_manager.colors.ACCENT, width=2)
        )
        self.equity_graph.addItem(equity_curve_item)
        
        # Imposta i limiti dell'asse Y per equity
        min_equity = min(equity_values)
        max_equity = max(equity_values)
        equity_range = max_equity - min_equity
        
        # Aggiungi un margine del 5%
        margin = equity_range * 0.05
        self.equity_graph.setYRange(min_equity - margin, max_equity + margin)
        
        # Prepara i dati per il grafico drawdown
        dd_dates = [item[0] for item in drawdown_curve]
        dd_values = [item[1] for item in drawdown_curve]
        
        # Crea l'asse X personalizzato per le date
        time_axis_dd = TimeAxisItem(orientation='bottom')
        time_axis_dd.set_timestamps(dd_dates)
        self.drawdown_graph.setAxisItems({'bottom': time_axis_dd})
        
        # Grafico drawdown
        dd_curve_item = pg.PlotDataItem(
            x=list(range(len(dd_values))),
            y=dd_values,
            pen=pg.mkPen(color=style_manager.colors.ERROR, width=2),
            fillLevel=0,
            brush=pg.mkBrush(color=QColor(style_manager.colors.ERROR).lighter(180))
        )
        self.drawdown_graph.addItem(dd_curve_item)
        
        # Imposta i limiti dell'asse Y per drawdown (invertito)
        max_dd = max(dd_values)
        self.drawdown_graph.setYRange(0, max_dd * 1.1)
        
        # Sincronizza i grafici sull'asse X
        self.equity_graph.setXRange(0, len(equity_values) - 1)
        self.drawdown_graph.setXRange(0, len(dd_values) - 1)