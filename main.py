import websocket
import json
import threading
import time
import numpy as np
import pandas_ta as ta
from datetime import datetime
import ssl
from collections import deque
import requests

class BOOM1000MTFAnalyzer:
    def __init__(self, token, app_id="88258", telegram_token=None, telegram_chat_id=None):
        # --- Configuración de Conexión ---
        self.ws_url = f"wss://ws.derivws.com/websockets/v3?app_id={app_id}"
        self.token = token
        self.ws = None; self.connected = False; self.authenticated = False
        
        # --- Configuración de Telegram ---
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.telegram_enabled = telegram_token is not None and telegram_chat_id is not None

        # --- 📈 MTF: Configuración de Timeframes ---
        self.symbol = "BOOM1000"
        self.ltf_interval_seconds = 60
        self.htf_interval_seconds = 300
        self.min_ltf_candles = 50
        self.min_htf_candles = 10

        # --- Parámetros de la Estrategia ---
        self.ema_fast_period = 9; self.ema_slow_period = 21; self.ema_trend_period = 50
        self.rsi_period = 14; self.atr_period = 14; self.volume_ema_period = 20
        self.htf_trend_ema_period = 21
        self.sl_atr_multiplier = 1.5; self.tp_atr_multiplier = 2.0

        # --- Almacenamiento de Datos ---
        self.ltf_candles = deque(maxlen=200)
        self.htf_candles = deque(maxlen=100)
        self.ticks_for_current_candle = []
        self.last_ltf_candle_timestamp = 0
        self.last_htf_candle_timestamp = 0
        self.new_ltf_candle_ready = False

        # --- Estado de Trading ---
        self.active_trade = None
        self.dominant_trend = "NEUTRAL"
        self.last_signal_time = 0; self.signal_cooldown = self.ltf_interval_seconds * 2

    def send_telegram_message(self, message):
        """Envía un mensaje a través de Telegram"""
        if not self.telegram_enabled:
            return False
            
        try:
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                "chat_id": self.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Error enviando mensaje a Telegram: {e}")
            return False

    def connect(self):
        print("🌐 Conectando a Deriv API...")
        self.ws = websocket.WebSocketApp(
            self.ws_url, 
            on_open=self.on_open, 
            on_message=self.on_message, 
            on_error=self.on_error, 
            on_close=self.on_close
        )
        wst = threading.Thread(
            target=self.ws.run_forever, 
            kwargs={'sslopt': {"cert_reqs": ssl.CERT_NONE}, 'ping_interval': 30, 'ping_timeout': 10}
        )
        wst.daemon = True
        wst.start()
        time.sleep(5)
        return self.connected

    def on_open(self, ws): 
        print("✅ Conexión abierta")
        self.connected = True
        ws.send(json.dumps({"authorize": self.token}))

    def on_close(self, ws, code, msg): 
        print("🔌 Conexión cerrada")
        self.connected = False
        self.authenticated = False

    def on_error(self, ws, error): 
        print(f"❌ Error WebSocket: {error}")

    def on_message(self, ws, message):
        data = json.loads(message)
        if "error" in data: 
            print(f"❌ Error: {data['error'].get('message', 'Error')}")
        elif "authorize" in data: 
            print("✅ Autenticación exitosa.")
            self.authenticated = True
            self.subscribe_to_ticks()
        elif "tick" in data: 
            self.handle_tick(data['tick'])

    def subscribe_to_ticks(self): 
        print(f"📊 Suscribiendo a ticks de {self.symbol}...")
        self.ws.send(json.dumps({"ticks": self.symbol, "subscribe": 1}))

    def handle_tick(self, tick):
        try:
            price = float(tick['quote'])
            timestamp = int(tick['epoch'])
            current_ltf_start_time = timestamp - (timestamp % self.ltf_interval_seconds)
            
            if self.last_ltf_candle_timestamp == 0: 
                self.last_ltf_candle_timestamp = current_ltf_start_time
            
            if current_ltf_start_time > self.last_ltf_candle_timestamp:
                self._finalize_ltf_candle()
                self.last_ltf_candle_timestamp = current_ltf_start_time
            
            self.ticks_for_current_candle.append(price)
        except Exception as e:
            print(f"❌ Error en handle_tick: {e}")

    def _finalize_ltf_candle(self):
        if not self.ticks_for_current_candle: 
            return
        
        prices = np.array(self.ticks_for_current_candle)
        candle = {
            'timestamp': self.last_ltf_candle_timestamp, 
            'open': prices[0], 
            'high': np.max(prices),
            'low': np.min(prices), 
            'close': prices[-1], 
            'volume': len(prices)
        }
        
        self.ltf_candles.append(candle)
        self.ticks_for_current_candle = []
        self.new_ltf_candle_ready = True

        # ✅ MEJORA: Añadir feedback de la recopilación inicial de velas
        if len(self.ltf_candles) < self.min_ltf_candles:
            print(f"\r⏳ Recopilando velas iniciales (1min): {len(self.ltf_candles)}/{self.min_ltf_candles}", end="")
        elif len(self.ltf_candles) == self.min_ltf_candles:
            print(f"\n✅ Recopilación de {self.min_ltf_candles} velas completa. Iniciando análisis de mercado...")

        if self.last_htf_candle_timestamp == 0:
            self.last_htf_candle_timestamp = candle['timestamp'] - (candle['timestamp'] % self.htf_interval_seconds)

        if candle['timestamp'] >= self.last_htf_candle_timestamp + self.htf_interval_seconds:
            self._build_and_analyze_htf_candle()
            self.last_htf_candle_timestamp += self.htf_interval_seconds

    def _build_and_analyze_htf_candle(self):
        num_ltf_in_htf = self.htf_interval_seconds // self.ltf_interval_seconds
        candles_for_htf = list(self.ltf_candles)[-num_ltf_in_htf:]
        if not candles_for_htf: 
            return

        htf_open = candles_for_htf[0]['open']
        htf_high = max(c['high'] for c in candles_for_htf)
        htf_low = min(c['low'] for c in candles_for_htf)
        htf_close = candles_for_htf[-1]['close']
        htf_volume = sum(c['volume'] for c in candles_for_htf)

        htf_candle = {
            'timestamp': self.last_htf_candle_timestamp, 
            'open': htf_open, 
            'high': htf_high,
            'low': htf_low, 
            'close': htf_close, 
            'volume': htf_volume
        }
        self.htf_candles.append(htf_candle)

        if len(self.htf_candles) >= self.min_htf_candles:
            # Convertir a DataFrame para pandas_ta
            import pandas as pd
            df_htf = pd.DataFrame(list(self.htf_candles))
            
            # Calcular EMA usando pandas_ta
            htf_ema = ta.ema(df_htf['close'], length=self.htf_trend_ema_period)
            
            old_trend = self.dominant_trend
            if df_htf['close'].iloc[-1] > htf_ema.iloc[-1]: 
                self.dominant_trend = "UP"
            elif df_htf['close'].iloc[-1] < htf_ema.iloc[-1]: 
                self.dominant_trend = "DOWN"
            else: 
                self.dominant_trend = "NEUTRAL"

            if self.dominant_trend != old_trend and self.dominant_trend != "NEUTRAL":
                trend_color = "\033[92m" if self.dominant_trend == "UP" else "\033[91m"
                print(f"\n{trend_color}📈 TENDENCIA DOMINANTE (5min) CAMBIÓ A: {self.dominant_trend}\033[0m")
                
                # Enviar notificación de cambio de tendencia a Telegram
                if self.telegram_enabled:
                    trend_emoji = "📈" if self.dominant_trend == "UP" else "📉"
                    message = f"{trend_emoji} <b>TENDENCIA CAMBIADA</b>\n\n"
                    message += f"<b>Par:</b> {self.symbol}\n"
                    message += f"<b>Timeframe:</b> 5min\n"
                    message += f"<b>Nueva Tendencia:</b> {self.dominant_trend}\n"
                    message += f"<b>Hora:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    self.send_telegram_message(message)

    def analyze_market(self):
        if len(self.ltf_candles) < self.min_ltf_candles or self.active_trade: 
            return

        # Convertir a arrays de numpy
        opens = np.array([c['open'] for c in self.ltf_candles], dtype=float)
        highs = np.array([c['high'] for c in self.ltf_candles], dtype=float)
        lows = np.array([c['low'] for c in self.ltf_candles], dtype=float)
        closes = np.array([c['close'] for c in self.ltf_candles], dtype=float)
        volumes = np.array([c['volume'] for c in self.ltf_candles], dtype=float)

        # Convertir a DataFrame para pandas_ta
        import pandas as pd
        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows, 
            'close': closes, 'volume': volumes
        })

        # Calcular indicadores con pandas_ta
        ema_fast = ta.ema(df['close'], length=self.ema_fast_period)
        ema_slow = ta.ema(df['close'], length=self.ema_slow_period)
        ema_trend = ta.ema(df['close'], length=self.ema_trend_period)
        rsi = ta.rsi(df['close'], length=self.rsi_period)
        atr = ta.atr(df['high'], df['low'], df['close'], length=self.atr_period)
        volume_ema = ta.ema(df['volume'], length=self.volume_ema_period)
        
        # Calcular patrón engulfing manualmente (pandas_ta no tiene CDLENGULFING)
        engulfing = np.zeros(len(df))
        for i in range(1, len(df)):
            prev_open, prev_close = df['open'].iloc[i-1], df['close'].iloc[i-1]
            curr_open, curr_close = df['open'].iloc[i], df['close'].iloc[i]
            
            # Bullish engulfing
            if (curr_close > curr_open and prev_close < prev_open and 
                curr_open < prev_close and curr_close > prev_open):
                engulfing[i] = 100
            # Bearish engulfing
            elif (curr_close < curr_open and prev_close > prev_open and 
                  curr_open > prev_close and curr_close < prev_open):
                engulfing[i] = -100

        last_close = closes[-1]
        last_atr = atr.iloc[-1] if not atr.empty else 0
        is_volume_spike = volumes[-1] > volume_ema.iloc[-1] * 1.2 if not volume_ema.empty else False
        is_bullish_engulfing = engulfing[-1] == 100
        is_bearish_engulfing = engulfing[-1] == -100

        signal = None
        reason = ""
        
        if time.time() - self.last_signal_time < self.signal_cooldown: 
            return

        if self.dominant_trend == "UP":
            is_uptrend_ltf = (not ema_fast.empty and not ema_slow.empty and not ema_trend.empty and
                             ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_slow.iloc[-1] > ema_trend.iloc[-1])
            is_ema_cross_up = (len(ema_fast) > 1 and len(ema_slow) > 1 and
                              ema_fast.iloc[-2] <= ema_slow.iloc[-2] and ema_fast.iloc[-1] > ema_slow.iloc[-1])
            
            if is_uptrend_ltf and is_ema_cross_up and is_volume_spike:
                signal = "BUY"
                reason = "Cruce de EMAs (LTF) con Tendencia (HTF)"
            if is_bullish_engulfing and is_uptrend_ltf and (not rsi.empty and rsi.iloc[-1] < 75) and is_volume_spike:
                signal = "BUY"
                reason = "Vela Envolvente (LTF) con Tendencia (HTF)"

        if self.dominant_trend == "DOWN":
            is_downtrend_ltf = (not ema_fast.empty and not ema_slow.empty and not ema_trend.empty and
                               ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_slow.iloc[-1] < ema_trend.iloc[-1])
            is_ema_cross_down = (len(ema_fast) > 1 and len(ema_slow) > 1 and
                                ema_fast.iloc[-2] >= ema_slow.iloc[-2] and ema_fast.iloc[-1] < ema_slow.iloc[-1])
            
            if is_downtrend_ltf and is_ema_cross_down and is_volume_spike:
                signal = "SELL"
                reason = "Cruce de EMAs (LTF) con Tendencia (HTF)"
            if is_bearish_engulfing and is_downtrend_ltf and (not rsi.empty and rsi.iloc[-1] > 25) and is_volume_spike:
                signal = "SELL"
                reason = "Vela Envolvente (LTF) con Tendencia (HTF)"

        if signal:
            self.last_signal_time = time.time()
            self.display_signal(signal, reason, last_close, last_atr, rsi.iloc[-1] if not rsi.empty else 50)
            sl, tp = self.calculate_tp_sl(signal, last_close, last_atr)
            self.active_trade = {"direction": signal, "entry": last_close, "sl": sl, "tp": tp}
            
            # Enviar señal a Telegram
            if self.telegram_enabled:
                self.send_signal_to_telegram(signal, reason, last_close, sl, tp, rsi.iloc[-1] if not rsi.empty else 50)

    def send_signal_to_telegram(self, signal, reason, price, sl, tp, rsi):
        """Envía una señal de trading a Telegram"""
        direction_emoji = "🟢" if signal == "BUY" else "🔴"
        message = f"{direction_emoji} <b>SEÑAL DE TRADING</b> {direction_emoji}\n\n"
        message += f"<b>Par:</b> {self.symbol}\n"
        message += f"<b>Dirección:</b> {signal}\n"
        message += f"<b>Precio:</b> {price:.2f}\n"
        message += f"<b>Take Profit:</b> {tp:.2f}\n"
        message += f"<b>Stop Loss:</b> {sl:.2f}\n"
        message += f"<b>RSI:</b> {rsi:.2f}\n"
        message += f"<b>Razón:</b> {reason}\n"
        message += f"<b>Tendencia HTF:</b> {self.dominant_trend}\n"
        message += f"<b>Hora:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        message += "#TradingSignal #BOOM1000"
        
        self.send_telegram_message(message)

    def check_active_trade(self):
        if not self.active_trade: 
            return
        
        last_price = self.ltf_candles[-1]['close'] if self.ltf_candles else 0
        if not last_price: 
            return
        
        trade = self.active_trade
        exit_reason = None
        
        if trade['direction'] == "BUY":
            if last_price >= trade['tp']: 
                exit_reason = "Take Profit"
            elif last_price <= trade['sl']: 
                exit_reason = "Stop Loss"
        elif trade['direction'] == "SELL":
            if last_price <= trade['tp']: 
                exit_reason = "Take Profit"
            elif last_price >= trade['sl']: 
                exit_reason = "Stop Loss"
                
        if exit_reason:
            print(f"\n🔔 CIERRE DE OPERACIÓN ({trade['direction']}): {exit_reason} en {last_price:.2f}")
            
            # Enviar notificación de cierre a Telegram
            if self.telegram_enabled:
                result_emoji = "✅" if exit_reason == "Take Profit" else "❌"
                message = f"{result_emoji} <b>OPERACIÓN CERRADA</b> {result_emoji}\n\n"
                message += f"<b>Par:</b> {self.symbol}\n"
                message += f"<b>Dirección:</b> {trade['direction']}\n"
                message += f"<b>Entrada:</b> {trade['entry']:.2f}\n"
                message += f"<b>Salida:</b> {last_price:.2f}\n"
                message += f"<b>Resultado:</b> {exit_reason}\n"
                
                if exit_reason == "Take Profit":
                    profit = last_price - trade['entry'] if trade['direction'] == "BUY" else trade['entry'] - last_price
                    message += f"<b>Ganancia:</b> {profit:.2f} puntos\n"
                
                message += f"<b>Hora:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                self.send_telegram_message(message)
                
            self.active_trade = None

    def calculate_tp_sl(self, direction, price, atr_value):
        if direction == "BUY": 
            return round(price - (atr_value * self.sl_atr_multiplier), 2), round(price + (atr_value * self.tp_atr_multiplier), 2)
        else: 
            return round(price + (atr_value * self.sl_atr_multiplier), 2), round(price - (atr_value * self.tp_atr_multiplier), 2)

    def display_signal(self, direction, reason, price, atr_value, rsi_value):
        sl, tp = self.calculate_tp_sl(direction, price, atr_value)
        color = "\033[92m" if direction == "BUY" else "\033[91m"
        reset = "\033[0m"
        
        print("\n" + "="*70)
        print(f"🎯 {color}SEÑAL MTF DETECTADA - {direction} (TENDENCIA 5MIN: {self.dominant_trend}){reset}")
        print("="*70)
        print(f"   🧠 Razón: {reason}")
        print(f"   💰 Precio: {price:.2f} | 🎯 TP: {tp:.2f} | 🛑 SL: {sl:.2f} | 📊 RSI: {rsi_value:.1f}")
        print("="*70)

    def run(self):
        print("\n" + "="*70)
        print("🤖 ANALIZADOR MTF BOOM 1000 v4.1 (Pandas TA)")
        print("="*70)
        print("💡 ESTRATEGIA: Tendencia en 5min (HTF) + Entradas en 1min (LTF)")
        
        if self.telegram_enabled:
            print("📱 NOTIFICACIONES: Telegram habilitado")
        else:
            print("📱 NOTIFICACIONES: Telegram deshabilitado")
        
        print("="*70)
        
        if self.connect():
            try:
                while self.connected:
                    if self.new_ltf_candle_ready:
                        if self.active_trade: 
                            self.check_active_trade()
                        else: 
                            self.analyze_market()
                        self.new_ltf_candle_ready = False
                    time.sleep(1)
            except KeyboardInterrupt: 
                print("\n🛑 Deteniendo analizador...")
        else: 
            print("❌ No se pudo conectar a Deriv")

if __name__ == "__main__":
    DEMO_TOKEN = "a1-m63zGttjKYP6vUq8SIJdmySH8d3Jc"
    
    # Configuración de Telegram (reemplaza con tus propios valores)
    TELEGRAM_BOT_TOKEN = "7868591681:AAGYeuSUwozg3xTi1zmxPx9gWRP2xsXP0Uc"
    TELEGRAM_CHAT_ID = "-1003028922957"
    
    analyzer = BOOM1000MTFAnalyzer(
        token=DEMO_TOKEN, 
        telegram_token=TELEGRAM_BOT_TOKEN, 
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    analyzer.run()