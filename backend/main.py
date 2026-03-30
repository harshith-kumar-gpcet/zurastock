import yfinance as yf
from curl_cffi import requests
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, Depends, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
from sqlalchemy.orm import Session
from .database import SessionLocal, engine, Base, User, Watchlist, Portfolio, init_db, get_db
import math
import pandas as pd
import ta
import asyncio
import json
print("INFO: ZuraStock Backend V2 (Package Fix) - Starting Up...")
from contextlib import asynccontextmanager
import numpy as np
from textblob import TextBlob
from .ml_engine import MLEngine
from .ml_assistant import MLAssistant
from datetime import datetime, time
import pytz

# Create a persistent session with curl_cffi to prevent rate limiting/blocking.
# Using 'chrome120' impersonation to mimic a modern browser.
session = requests.Session(impersonate="chrome120")

def is_market_open():
    """Checks if the Indian stock market is currently open."""
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz)
    
    # Weekend check
    if now.weekday() >= 5: # 5=Saturday, 6=Sunday
        return False
        
    market_open = time(9, 15)
    market_close = time(15, 30)
    current_time = now.time()
    
    return market_open <= current_time <= market_close

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        msg_str = json.dumps(message)
        for connection in self.active_connections:
            try:
                await connection.send_text(msg_str)
            except Exception:
                pass

manager = ConnectionManager()

# --- Utility Functions ---
def format_volume(vol):
    try:
        if math.isnan(vol): return "0"
        vol = int(vol)
        if vol >= 1000000: return f"{vol/1000000:.1f}M"
        if vol >= 1000: return f"{vol/1000:.1f}K"
        return str(vol)
    except: return "0"

def map_news_item(item):
    """Helper to handle new and old yfinance news structures."""
    # New structure (Nested under 'content')
    content = item.get("content")
    if content:
        title = content.get("title") or content.get("summary") or "Market Update"
        # Truncate title if extremely long
        if len(title) > 200: title = title[:197] + "..."
        
        provider = content.get("provider", {})
        publisher = provider.get("displayName") or "Market News"
        
        url = content.get("clickThroughUrl", {}).get("url") or \
              content.get("canonicalUrl", {}).get("url") or "#"
              
        thumbnail = None
        if content.get("thumbnail") and content["thumbnail"].get("resolutions"):
            thumbnail = content["thumbnail"]["resolutions"][0].get("url")
            
        return {
            "id": content.get("id", str(hash(title))),
            "title": title,
            "link": url,
            "publisher": publisher,
            "image": thumbnail,
            "desc": content.get("summary", title)
        }
    
    # Old structure (directly on root)
    title = item.get("title") or item.get("headline") or "Market Update"
    publisher = item.get("publisher") or "Market News"
    return {
        "id": item.get("uuid", str(hash(title))),
        "title": title,
        "link": item.get("link") or "#",
        "publisher": publisher,
        "image": None,
        "desc": title
    }

# --- Global Cache & Market Status ---
market_cache = {}
indices_cache = {}
top_picks_cache = []
top_news_cache = []
fundamentals_cache = {}
cache_lock = asyncio.Lock()

def init_cache_metadata():
    for sym, meta in STOCK_METADATA.items():
        market_cache[sym] = {
            "symbol": sym,
            "name": meta["name"],
            "sector": meta["sector"],
            "cap": meta.get("cap", "N/A"),
            "price": 0.0,
            "change": 0.0,
            "changePercent": 0.0,
            "volume": "0",
            "open": 0.0,
            "high": 0.0,
            "low": 0.0
        }

# --- Background Market Broadcast ---
async def broadcast_market_data():
    """Continuously populates cache and broadcasts to WS clients"""
    global market_cache, indices_cache
    
    # Track the last time we fetched data when market was closed
    last_closed_fetch = None
    
    while True:
        try:
            market_open = is_market_open()
            
            # Logic: If market is closed, only fetch once an hour.
            # CRITICAL: We must fetch at least once on startup (if prices are all 0).
            has_real_data = any(d.get("price", 0) > 0 for d in market_cache.values())
            
            if not market_open and last_closed_fetch and (datetime.now() - last_closed_fetch).total_seconds() < 3600:
                if has_real_data:
                    await asyncio.sleep(60)
                    continue
                    
            if not market_open:
                last_closed_fetch = datetime.now()

            # 1. Update Market Indices (More robust fetching)
            indices_map = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BANK NIFTY": "^NSEBANK"}
            for name, ticker in indices_map.items():
                try:
                    ticker_obj = yf.Ticker(ticker, session=session)
                    hist = ticker_obj.history(period="2d")
                    if not hist.empty:
                        last = hist.iloc[-1]
                        prev = hist.iloc[-2] if len(hist) > 1 else last
                        price = float(last['Close'])
                        change = price - float(prev['Close'])
                        async with cache_lock:
                            indices_cache[name] = {
                                "price": price,
                                "change": change,
                                "percent": (change / float(prev['Close'])) * 100 if prev['Close'] else 0
                            }
                except Exception as e:
                    print(f"Warning: Failed to fetch index {name}: {e}")
                    continue

            # NEW: Broadcast indices immediately before processing stocks
            if manager.active_connections and indices_cache:
                await manager.broadcast({"indices": indices_cache, "stocks": market_cache})

            # 2. Update Stocks in Chunks
            all_symbols = list(DEFAULT_SYMBOLS.items())
            chunk_size = 25
            
            for i in range(0, len(all_symbols), chunk_size):
                chunk = dict(all_symbols[i:i + chunk_size])
                tickers_list = list(chunk.values())
                
                try:
                    # Using threads=False for stability with curl_cffi session
                    data = yf.download(tickers_list, period="5d", group_by="ticker", progress=False, threads=False, session=session)
                    
                    if data is None or data.empty:
                        # Non-fatal: if chunk fails, just log and continue to next chunk
                        print(f"Warning: No data for chunk: {tickers_list}")
                        continue

                    async with cache_lock:
                        for base, ticker in chunk.items():
                            meta = STOCK_METADATA.get(base)
                            if not meta: continue
                            
                            try:
                                # Standardize data access with robust MultiIndex handling
                                df_ticker = None
                                if len(tickers_list) > 1:
                                    if ticker in data.columns.levels[0]:
                                        df_ticker = data[ticker].dropna()
                                else:
                                    df_ticker = data.dropna()
                                    
                                if df_ticker is None or df_ticker.empty:
                                    continue
                                
                                last_row = df_ticker.iloc[-1]
                                prev_row = df_ticker.iloc[-2] if len(df_ticker) > 1 else last_row
                                price = float(last_row['Close'])
                                prev_close = float(prev_row['Close'])
                                
                                market_cache[base] = {
                                    "symbol": base,
                                    "name": meta["name"],
                                    "sector": meta["sector"],
                                    "cap": meta.get("cap", "N/A"),
                                    "ceo": meta.get("ceo", "N/A"),
                                    "price": price,
                                    "change": price - prev_close,
                                    "changePercent": ((price - prev_close) / prev_close) * 100 if prev_close else 0,
                                    "volume": format_volume(last_row['Volume']),
                                    "open": float(last_row['Open']),
                                    "high": float(last_row['High']),
                                    "low": float(last_row['Low'])
                                }
                            except Exception as inner_e:
                                # Prevent single ticker Failure from killing the loop
                                # print(f"Ticker processing error ({base}): {inner_e}")
                                pass
                        
                        print(f"INFO: Successfully updated chunk: {tickers_list[:3]}... ({len(tickers_list)} symbols)")
                    
                    # Broadcast indices + stocks periodically
                    if manager.active_connections:
                        await manager.broadcast({"indices": indices_cache, "stocks": market_cache})

                except Exception as chunk_e:
                    # Catch and log download errors (like 404s/timeouts) for the chunk
                    if "404" in str(chunk_e):
                        print(f"Known Issue: Chunk download 404'd (Some symbols might be delisted).")
                    else:
                        print(f"Chunk Download Error: {chunk_e}")
                
                # Brief pause between chunks to respect API/resource limits
                await asyncio.sleep(2)
            
            # 10-minute rule when market is open
            if is_market_open():
                await asyncio.sleep(600)  # 10 minutes
            else:
                await asyncio.sleep(1800) # 30 minutes when closed

        except Exception as global_e:
            print(f"Global Broadcaster Error: {global_e}")
            await asyncio.sleep(60) # Longer sleep on global failure

# --- Background Top Picks Calculator ---
async def calculate_top_picks_bg():
    global top_picks_cache, market_cache
    
    # Wait for initial market data to populate
    await asyncio.sleep(10)
    
    while True:
        try:
            picks = []
            async with cache_lock:
                all_symbols = list(market_cache.keys())
            
            if not all_symbols:
                await asyncio.sleep(30)
                continue
                
            # Limit to first 6 for speed and run in parallel
            symbols_to_analyze = all_symbols[:6]
            tasks = [get_analysis(s) for s in symbols_to_analyze]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, analysis in enumerate(results):
                if isinstance(analysis, dict) and analysis.get("confidence", 0) > 0:
                    symbol = symbols_to_analyze[i]
                    picks.append({
                        "symbol": symbol,
                        "name": market_cache.get(symbol, {}).get("name", symbol),
                        "price": market_cache.get(symbol, {}).get("price", 0),
                        "change": market_cache.get(symbol, {}).get("changePercent", 0),
                        "confidence": analysis["confidence"],
                        "signal": analysis["signal"]
                    })
            
            # Sort by confidence and take top 4
            picks.sort(key=lambda x: x["confidence"], reverse=True)
            
            async with cache_lock:
                top_picks_cache = picks[:4]
                
            if is_market_open():
                # Recalculate every 15 minutes during market hours
                await asyncio.sleep(900)
            else:
                # Recalculate once an hour when closed
                await asyncio.sleep(3600)
            
        except Exception as e:
            print(f"Top Picks BG Error: {e}")
            await asyncio.sleep(60)

# --- Background Top News Fetcher ---
async def fetch_top_news_bg():
    global top_news_cache
    
    # Wait for startup
    await asyncio.sleep(15)
    
    top_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "TATACONSUM"]
    
    while True:
        try:
            all_news = []
            
            async def fetch_news(symbol):
                try:
                    ticker_str = f"{symbol}.NS"
                    stock = yf.Ticker(ticker_str)
                    news_items = stock.news[:2]
                    results = []
                    for item in news_items:
                        data = map_news_item(item)
                        sentiment = TextBlob(data["title"]).sentiment.polarity
                        results.append({
                            "id": data["id"],
                            "title": data["title"],
                            "link": data["link"],
                            "publisher": data["publisher"],
                            "source": data["publisher"],
                            "time": "Recent",
                            "image": data["image"] or "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&q=80&w=400",
                            "desc": data["desc"],
                            "sentiment": "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
                        })
                    return results
                except Exception as e:
                    print(f"Fetch news error for {symbol}: {e}")
                    return []

            tasks = [fetch_news(s) for s in top_stocks]
            news_batches = await asyncio.gather(*tasks)
            for batch in news_batches:
                all_news.extend(batch)
                    
            if all_news:
                import random
                random.shuffle(all_news)
                async with cache_lock:
                    top_news_cache = all_news[:10]
            
            if is_market_open():
                # Fetch news every 30 minutes during market hours
                await asyncio.sleep(1800)
            else:
                # Fetch news every 2 hours when closed
                await asyncio.sleep(7200)
            
        except Exception as e:
            print(f"Top News BG Error: {e}")
            await asyncio.sleep(60)

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize cache with metadata first
    init_cache_metadata()
    # Initialize DB on startup
    init_db()
    # Pre-populate market_cache with metadata to avoid empty dashboard on cold start
    print("INFO: Pre-populating market cache with metadata...")
    async with cache_lock:
        for base, meta in STOCK_METADATA.items():
            if base not in market_cache:
                market_cache[base] = {
                    "symbol": base,
                    "name": meta["name"],
                    "sector": meta["sector"],
                    "cap": meta.get("cap", "N/A"),
                    "ceo": meta.get("ceo", "N/A"),
                    "price": 0.0,
                    "change": 0.0,
                    "changePercent": 0.0,
                    "volume": "0",
                    "open": 0.0,
                    "high": 0.0,
                    "low": 0.0
                }
    print(f"INFO: Cache pre-populated with {len(market_cache)} symbols.")

    asyncio.create_task(broadcast_market_data())
    asyncio.create_task(calculate_top_picks_bg())
    asyncio.create_task(fetch_top_news_bg())
    yield

app = FastAPI(lifespan=lifespan)

# --- Endpoints ---
@app.get("/api/stocks")
async def get_stocks(symbols: str = Query(None), sector: str = Query(None)):
    async with cache_lock:
        if symbols:
            result = {}
            symbol_list = [s.strip().upper() for s in symbols.split(",")]
            for s in symbol_list:
                if s in market_cache:
                    result[s] = market_cache[s]
                else:
                    # Dynamic fetch logic (keeping it inside symbols block)
                    try:
                        ticker_str = s if "." in s else f"{s}.NS"
                        ticker_obj = yf.Ticker(ticker_str)
                        hist = ticker_obj.history(period="2d")
                        if not hist.empty:
                            last = hist.iloc[-1]
                            prev = hist.iloc[-2] if len(hist) > 1 else last
                            price = float(last['Close'])
                            change = price - float(prev['Close'])
                            stock_data = {
                                "symbol": s,
                                "name": s,
                                "sector": "Other",
                                "price": price,
                                "change": change,
                                "changePercent": (change / float(prev['Close'])) * 100 if prev['Close'] else 0,
                                "volume": format_volume(last['Volume']),
                                "open": float(last['Open']),
                                "high": float(last['High']),
                                "low": float(last['Low'])
                            }
                            market_cache[s] = stock_data
                            result[s] = stock_data
                    except: pass
            return result
            
        # If no specific symbols, return list (filtered by sector if provided)
        stocks_list = []
        for sym, data in market_cache.items():
            # Inject symbol if not present
            data["symbol"] = sym
            if not sector or data.get("sector") == sector:
                stocks_list.append(data)
        return stocks_list

# --- DB Backed Endpoints ---

@app.get("/api/user")
async def get_current_user(db: Session = Depends(get_db)):
    # For demo, we just use the default user
    user = db.query(User).filter(User.username == "demo_user").first()
    return {"id": user.id, "username": user.username}

@app.get("/api/db-watchlist")
async def get_db_watchlist(db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == "demo_user").first()
    items = db.query(Watchlist).filter(Watchlist.user_id == user.id).all()
    return [item.symbol for item in items]

@app.post("/api/db-watchlist/toggle")
async def toggle_db_watchlist(symbol: str = Body(..., embed=True), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == "demo_user").first()
    symbol = symbol.upper()
    existing = db.query(Watchlist).filter(Watchlist.user_id == user.id, Watchlist.symbol == symbol).first()
    
    if existing:
        db.delete(existing)
        db.commit()
        return {"status": "removed", "symbol": symbol}
    else:
        new_item = Watchlist(user_id=user.id, symbol=symbol)
        db.add(new_item)
        db.commit()
        return {"status": "added", "symbol": symbol}

@app.get("/api/db-portfolio")
async def get_db_portfolio(db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == "demo_user").first()
    items = db.query(Portfolio).filter(Portfolio.user_id == user.id).all()
    return [{"symbol": i.symbol, "quantity": i.quantity, "avg_price": i.avg_price} for i in items]

@app.post("/api/db-portfolio/add")
async def add_to_db_portfolio(symbol: str = Body(...), quantity: int = Body(...), price: float = Body(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == "demo_user").first()
    symbol = symbol.upper()
    
    # Simple logic: if exists, update avg price and qty, else create
    existing = db.query(Portfolio).filter(Portfolio.user_id == user.id, Portfolio.symbol == symbol).first()
    if existing:
        total_qty = existing.quantity + quantity
        if total_qty <= 0:
            db.delete(existing)
        else:
            existing.avg_price = ((existing.avg_price * existing.quantity) + (price * quantity)) / total_qty
            existing.quantity = total_qty
    else:
        if quantity > 0:
            new_item = Portfolio(user_id=user.id, symbol=symbol, quantity=quantity, avg_price=price)
            db.add(new_item)
            
    db.commit()
    return {"status": "success"}

@app.get("/api/ohlc")
async def get_ohlc(symbol: str, period: str = "1mo"):
    """Returns OHLC data formatted for TradingView Lightweight Charts."""
    # Map periods to yfinance compatible strings
    period_map = {
        "1M": "1mo",
        "6M": "6mo",
        "1Y": "1y",
        "1m": "1mo",
        "6m": "6mo",
        "1y": "1y"
    }
    yf_period = period_map.get(period, period)
    
    try:
        ticker_str = symbol if "." in symbol else f"{symbol}.NS"
        df = await asyncio.to_thread(yf.download, ticker_str, period=yf_period, progress=False, session=session, threads=False)
        
        if df.empty:
            return []
            
        # Robust MultiIndex cleaning
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        ohlc = []
        for index, row in df.iterrows():
            ohlc.append({
                "time": int(index.timestamp()),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
        return ohlc
    except Exception as e:
        print(f"OHLC Error for {symbol}: {e}")
        return []

@app.get("/api/news")
async def get_news(symbol: str):
    """Fetches news and calculates sentiment mood."""
    try:
        ticker_str = symbol if "." in symbol else f"{symbol}.NS"
        stock = yf.Ticker(ticker_str)
        news_items = stock.news[:5] # Get latest 5
        
        processed_news = []
        total_polarity = 0
        
        for item in news_items:
            data = map_news_item(item)
            sentiment = TextBlob(data["title"]).sentiment.polarity
            total_polarity += sentiment
            
            processed_news.append({
                "title": data["title"],
                "link": data["link"],
                "publisher": data["publisher"],
                "sentiment": "Positive" if sentiment > 0.05 else "Negative" if sentiment < -0.05 else "Neutral"
            })
            
        avg_sentiment = total_polarity / len(news_items) if news_items else 0
        mood = "BULLISH" if avg_sentiment > 0.05 else "BEARISH" if avg_sentiment < -0.05 else "NEUTRAL"
        
        return {
            "mood": mood,
            "score": round(float(avg_sentiment), 2),
            "news": processed_news
        }
    except Exception as e:
        return {"error": str(e), "mood": "NEUTRAL", "news": []}

@app.get("/api/top-news")
async def get_top_news():
    """Returns pre-fetched market-wide news for top stocks."""
    async with cache_lock:
        return top_news_cache

@app.get("/api/top-picks")
async def get_top_picks():
    """Returns top 4 stocks with highest AI confidence."""
    async with cache_lock:
        return top_picks_cache

@app.get("/api/history")
async def get_history(symbol: str, period: str = "1mo"):
    """Fetches historical price data for charting."""
    try:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        # Map frontend periods to yfinance periods
        period_map = {"1M": "1mo", "6M": "6mo", "1Y": "1y"}
        yf_period = period_map.get(period.upper(), "1mo")
        
        # Download data
        df = await asyncio.to_thread(yf.download, ticker, period=yf_period, interval="1d", progress=False, session=session, threads=False)
        if df.empty:
            return {"labels": [], "data": []}
            
        # Robust MultiIndex cleaning
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Sampling for performance (max 100 points for 1Y)
        if len(df) > 100:
            step = len(df) // 100
            df = df.iloc[::step]
            
        labels = [d.strftime('%d %b') for d in df.index]
        prices_series = df['Close'].squeeze()
        if isinstance(prices_series, pd.DataFrame): prices_series = prices_series.iloc[:, 0]
        prices = [round(float(p), 2) for p in prices_series]
        
        return {
            "labels": labels,
            "data": prices,
            "period": period.upper()
        }
    except Exception as e:
        print(f"History Fetch Error: {e}")
        return {"error": str(e), "labels": [], "data": []}

@app.get("/api/forecast")
async def get_forecast(symbol: str):
    """Provides a 7-day ML-based price probability forecast."""
    try:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        df = await asyncio.to_thread(yf.download, ticker, period="3mo", interval="1d", progress=False, session=session, threads=False)
        if df.empty: return {"error": "No data"}
        
        # Robust MultiIndex cleaning
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        prices = df['Close'].squeeze()
        if isinstance(prices, pd.DataFrame): prices = prices.iloc[:, 0]
        prices = prices.dropna().astype(float).tolist()
        
        volumes = df['Volume'].squeeze()
        if isinstance(volumes, pd.DataFrame): volumes = volumes.iloc[:, 0]
        volumes = volumes.dropna().astype(float).tolist()
        
        if not prices: return {"error": "Insufficient price data"}
        
        forecast = MLEngine.generate_forecast(prices)
        anomaly = MLEngine.detect_anomalies(prices, volumes)
        
        return {
            "symbol": symbol,
            "forecast": forecast,
            "anomaly_detected": anomaly,
            "current_price": round(prices[-1], 2)
        }
    except Exception as e:
        print(f"ML Forecast Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def fetch_fundamentals_sync(symbol: str):
    try:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        stock = yf.Ticker(ticker, session=session)
        hist = stock.history(period="1d")
        info = stock.info
        fundamentals = {
            "marketCap": info.get("marketCap"), "peRatio": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"), "eps": info.get("trailingEps"),
            "dividendYield": info.get("dividendYield", 0) * 100 if info.get("dividendYield") else 0,
            "debtToEquity": info.get("debtToEquity"), "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"), "priceToBook": info.get("priceToBook"),
            "roe": info.get("returnOnEquity", 0) * 100 if info.get("returnOnEquity") else 0,
        }
        income_stmt = stock.income_stmt
        growth_data = []
        if not income_stmt.empty:
            df = income_stmt.T.sort_index().tail(4)
            for date, row in df.iterrows():
                growth_data.append({"date": date.strftime('%Y'), "revenue": float(row.get('Total Revenue', 0)), "netIncome": float(row.get('Net Income', 0))})
        return {
            "fundamentals": fundamentals, 
            "growth": growth_data,
            "description": info.get("longBusinessSummary", "No description available for this company."),
            "country": info.get("country", "IN"),
            "employees": info.get("fullTimeEmployees", "—"),
        }
    except Exception as e: return {"error": str(e)}

@app.get("/api/fundamentals")
async def get_fundamentals(symbol: str):
    if symbol in fundamentals_cache:
        return fundamentals_cache[symbol]
    
    res = await asyncio.to_thread(fetch_fundamentals_sync, symbol)
    if "error" not in res:
        fundamentals_cache[symbol] = res
    return res

@app.get("/api/analysis")
async def get_analysis(symbol: str, period: str = "1M"):
    try:
        ticker = symbol if "." in symbol else f"{symbol}.NS"
        
        # 1. Fetch Technical Data
        historical = yf.download(ticker, period="6mo", progress=False, session=session, threads=False)
        if historical.empty: return {"signal": "HOLD", "error": "No data"}
        if isinstance(historical.columns, pd.MultiIndex): historical.columns = historical.columns.get_level_values(0)
        close_prices = historical['Close'].squeeze()
        
        rsi = ta.momentum.rsi(close_prices, window=14).iloc[-1]
        sma_20 = ta.trend.sma_indicator(close_prices, window=20).iloc[-1]
        sma_50 = ta.trend.sma_indicator(close_prices, window=50).iloc[-1]
        current_price = float(close_prices.iloc[-1])
        
        # 2. Fetch Sentiment (Internal call logic)
        news_data = await get_news(symbol)
        sentiment_score = news_data.get("score", 0)
        
        # 3. Fetch Fundamental Health
        fund_response = await get_fundamentals(symbol)
        fundamentals = fund_response.get("fundamentals", {})
        
        # 4. RUN ML ENSEMBLE
        technicals = {"rsi": rsi, "sma_20": sma_20, "price": current_price}
        ml_result = MLEngine.get_ensemble_signal(technicals, sentiment_score, fundamentals)
        
        # 5. Fetch Historical Price Data for Chart
        history_response = await get_history(symbol, period)
        history_data = []
        if history_response.get("labels") and history_response.get("data"):
            history_data = [{"date": label, "close": price} for label, price in zip(history_response["labels"], history_response["data"])]
        
        # 6. Fetch Forecast Data
        forecast_response = await get_forecast(symbol)
        forecast_data = forecast_response.get("forecast", [])
        
        # Determine mood for AI assistant
        mood = "BULLISH" if ml_result["signal"] == "BUY" else "BEARISH" if ml_result["signal"] == "SELL" else "NEUTRAL"
        
        # Merge results for frontend
        return {
            "signal": ml_result["signal"],
            "confidence": ml_result["confidence"],
            "weights": ml_result["weights"],
            "insights": ml_result.get("insights", {"pros": [], "cons": []}),
            "explanation": f"AI Ensemble weighted {ml_result['signal']} signal based on multi-modal analysis.",
            "rsi": round(float(rsi), 2) if isinstance(rsi, (int, float, np.number)) else 50,
            "sma20": round(float(sma_20), 2) if isinstance(sma_20, (int, float, np.number)) else 0,
            "sma50": round(float(sma_50), 2) if isinstance(sma_50, (int, float, np.number)) else 0,
            "history": history_data,
            "forecast": forecast_data,
            "anomaly": forecast_response.get("anomaly_detected", False),
            "mood": mood
        }
    except Exception as e: 
        print(f"Analysis Error: {e}")
        import traceback
        traceback.print_exc()
        return {"signal": "HOLD", "reasoning": str(e), "confidence": 0}

@app.get("/api/ai-greeting")
async def get_ai_greeting():
    return {"message": MLAssistant.get_greeting()}

@app.get("/api/market-sentiment")
async def get_market_sentiment():
    """AI analysis of overall market sentiment based on top symbols."""
    async with cache_lock:
        if not market_cache:
            return {"sentiment": "Neutral", "score": 50, "outlook": "Waiting for data...", "color": "var(--muted)"}
        
        pos_count: int = 0
        total_count: int = 0
        for s, d in market_cache.items():
            change = d.get("changePercent", 0)
            if isinstance(change, (int, float)) and change > 0:
                pos_count += 1
            total_count += 1
            
        ratio = (pos_count / total_count) * 100 if total_count > 0 else 50
        
        if ratio > 65:
            return {"sentiment": "Strong Bullish", "score": round(ratio), "outlook": "Strong buying momentum detected across sectors.", "color": "var(--green)"}
        elif ratio < 35:
            return {"sentiment": "Bearish", "score": round(ratio), "outlook": "Caution: Selling pressure dominant in major indexes.", "color": "var(--red)"}
        else:
            return {"sentiment": "Neutral", "score": round(ratio), "outlook": "Market consolidating. Awaiting clear breakout signals.", "color": "var(--orange)"}

@app.post("/api/ai-chat")
async def ai_chat(data: dict = Body(...)):
    query = data.get("query", "")
    symbol = data.get("symbol", "RELIANCE").upper()
    
    # Get current context for analysis
    if symbol == "THE MARKET":
        # Provide general market context
        sentiment_data = await get_market_sentiment()
        context = {
            "symbol": "the market",
            "price": 0,
            "changePercent": 0,
            "signal": sentiment_data["sentiment"],
            "confidence": sentiment_data["score"],
            "insights": {"pros": [sentiment_data["outlook"]], "cons": []},
            "sentiment": sentiment_data["sentiment"].upper()
        }
    else:
        analysis = await get_analysis(symbol)
        context = {
            "symbol": symbol,
            "price": market_cache.get(symbol, {}).get("price", 0),
            "changePercent": market_cache.get(symbol, {}).get("changePercent", 0),
            "signal": analysis.get("signal", "HOLD"),
            "confidence": analysis.get("confidence", 0),
            "insights": analysis.get("insights", {"pros": [], "cons": []}),
            "sentiment": analysis.get("mood", "NEUTRAL")
        }
    
    response = MLAssistant.generate_response(query, context)
    return {"response": response, "symbol": symbol}

@app.post("/api/generate-portfolio")
async def generate_portfolio(data: dict = Body(...)):
    """
    AI-powered portfolio builder that optimizes asset allocation
    based on risk tolerance and capital.
    """
    amount = float(data.get("amount", 100000))
    risk = data.get("risk", "medium").lower()
    horizon = data.get("horizon", "medium").lower()
    
    # Select stocks based on risk profile
    if risk == "low":
        selected_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
        strategy = "Conservative Blue-Chip"
    elif risk == "high":
        selected_symbols = ["BAJFINANCE", "ZOMATO", "ICICIBANK", "SBIN", "RELIANCE"]
        strategy = "Aggressive Momentum"
    else:
        selected_symbols = ["RELIANCE", "TCS", "ICICIBANK", "INFY", "SBIN"]
        strategy = "Balanced Growth Growth"
    
    # Strategic Asset Allocation
    allocations = [0.35, 0.25, 0.20, 0.15, 0.05]
    portfolio = []
    total_invested = 0
    sectors = set()
    
    # Sector mapping (Static for now, can be dynamic with yfinance)
    sector_map = {
        "RELIANCE": "Energy", "TCS": "IT", "INFY": "IT", "HDFCBANK": "Finance", 
        "ICICIBANK": "Finance", "SBIN": "Finance", "ITC": "FMCG", "ZOMATO": "Tech",
        "BAJFINANCE": "Finance", "LART": "Infrastructure"
    }
    
    # Normalized Volatility (1-100)
    vol_map = {
        "RELIANCE": 15, "TCS": 12, "INFY": 18, "HDFCBANK": 14, 
        "ICICIBANK": 22, "SBIN": 25, "ITC": 10, "ZOMATO": 45,
        "BAJFINANCE": 35, "LART": 20
    }

    weighted_vol = 0
    
    for i, symbol in enumerate(selected_symbols):
        price = market_cache.get(symbol, {}).get("price", 2000)
        allocation_pct = allocations[i]
        target_amount = amount * allocation_pct
        shares = int(target_amount // price)
        
        if shares > 0:
            actual_amount = shares * price
            sector = sector_map.get(symbol, "Misc")
            portfolio.append({
                "symbol": symbol,
                "name": STOCK_METADATA.get(symbol, symbol),
                "price": price,
                "shares": shares,
                "allocation": f"{allocation_pct*100}%",
                "amount": actual_amount,
                "sector": sector
            })
            total_invested += actual_amount
            sectors.add(sector)
            weighted_vol += vol_map.get(symbol, 20) * allocation_pct

    # Professional Risk Metrics
    risk_metrics = {
        "volatility": round(weighted_vol, 1),
        "diversityScore": min(100, len(sectors) * 25), 
        "concentration": round(sum(allocations[:2]) * 100, 1),
        "sectors": list(sectors)
    }

    return {
        "portfolio": portfolio,
        "totalInvested": total_invested,
        "surplus": amount - total_invested,
        "strategy": strategy,
        "riskLevel": risk.capitalize(),
        "horizon": horizon.capitalize(),
        "analytics": risk_metrics
    }

@app.get("/api/search")
async def search_stocks(q: str):
    """Instant search autocomplete endpoint."""
    query = q.upper()
    results = []
    
    for symbol, yf_ticker in DEFAULT_SYMBOLS.items():
        meta = STOCK_METADATA.get(symbol, {"name": symbol, "sector": "Market"})
        if query in symbol or query in meta["name"].upper():
            results.append({
                "symbol": symbol,
                "name": meta["name"],
                "sector": meta["sector"]
            })
            if len(results) >= 8: # Limit to top 8 responses
                break
                
    return results

@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        async with cache_lock:
            await websocket.send_text(json.dumps({"indices": indices_cache, "stocks": market_cache}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- Standard Data ---
DEFAULT_SYMBOLS = {
    # Banking (16)
    "HDFCBANK": "HDFCBANK.NS", "ICICIBANK": "ICICIBANK.NS", "SBIN": "SBIN.NS", "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS", "INDUSINDBK": "INDUSINDBK.NS", "PNB": "PNB.NS", "BANKBARODA": "BANKBARODA.NS",
    "FEDERALBNK": "FEDERALBNK.NS", "IDFCFIRSTB": "IDFCFIRSTB.NS", "CANBK": "CANBK.NS", "AUBANK": "AUBANK.NS",
    "BANDHANBNK": "BANDHANBNK.NS", "IDBI": "IDBI.NS", "YESBANK": "YESBANK.NS", "RBLBANK": "RBLBANK.NS",

    # IT Services (15)
    "TCS": "TCS.NS", "INFY": "INFY.NS", "HCLTECH": "HCLTECH.NS", "WIPRO": "WIPRO.NS", "LTIM": "LTIM.NS",
    "TECHM": "TECHM.NS", "COFORGE": "COFORGE.NS", "PERSISTENT": "PERSISTENT.NS", "MPHASIS": "MPHASIS.NS",
    "KPITTECH": "KPITTECH.NS", "TATAELXSI": "TATAELXSI.NS", "LTTS": "LTTS.NS", "OFSS": "OFSS.NS",
    "BSOFT": "BSOFT.NS", "CYIENT": "CYIENT.NS",

    # Energy (15)
    "RELIANCE": "RELIANCE.NS", "ONGC": "ONGC.NS", "BPCL": "BPCL.NS", "IOC": "IOC.NS", "GAIL": "GAIL.NS",
    "NTPC": "NTPC.NS", "POWERGRID": "POWERGRID.NS", "ADANIGREEN": "ADANIGREEN.NS", "ADANIENSOL": "ADANIENSOL.NS",
    "TATAPOWER": "TATAPOWER.NS", "JSWENERGY": "JSWENERGY.NS", "NHPC": "NHPC.NS", "OIL": "OIL.NS",
    "PETRONET": "PETRONET.NS", "COALINDIA": "COALINDIA.NS",

    # Automobile (15)
    "MARUTI": "MARUTI.NS", "TATAMOTORS": "TATAMOTORS.NS", "M&M": "M&M.NS", "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "EICHERMOT": "EICHERMOT.NS", "HEROMOTOCO": "HEROMOTOCO.NS", "TVSMOTOR": "TVSMOTOR.NS", "ASHOKLEY": "ASHOKLEY.NS",
    "BALKRISIND": "BALKRISIND.NS", "MRF": "MRF.NS", "APOLLOTYRE": "APOLLOTYRE.NS", "BHARATFORG": "BHARATFORG.NS",
    "SONACOMS": "SONACOMS.NS", "TIINDIA": "TIINDIA.NS", "BOSCHLTD": "BOSCHLTD.NS",

    # Consumer Goods (15)
    "ITC": "ITC.NS", "HINDUNILVR": "HINDUNILVR.NS", "NESTLEIND": "NESTLEIND.NS", "BRITANNIA": "BRITANNIA.NS",
    "VBL": "VBL.NS", "TATACONSUM": "TATACONSUM.NS", "GODREJCP": "GODREJCP.NS", "DABUR": "DABUR.NS",
    "MARICO": "MARICO.NS", "COLPAL": "COLPAL.NS", "PGHH": "PGHH.NS", "BALRAMCHIN": "BALRAMCHIN.NS",
    "REDINGTON": "REDINGTON.NS", "UNITDSPR": "UNITDSPR.NS", "RADICO": "RADICO.NS",

    # Financial Services (15)
    "BAJFINANCE": "BAJFINANCE.NS", "BAJAJFINSV": "BAJAJFINSV.NS", "CHOLAFIN": "CHOLAFIN.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS", "MUTHOOTFIN": "MUTHOOTFIN.NS", "M&MFIN": "M&MFIN.NS",
    "RECLTD": "RECLTD.NS", "PFC": "PFC.NS", "L&TFH": "L&TFH.NS", "LICHSGFIN": "LICHSGFIN.NS",
    "ABCAPITAL": "ABCAPITAL.NS", "HDFCLIFE": "HDFCLIFE.NS", "SBILIFE": "SBILIFE.NS",
    "ICICIPRULI": "ICICIPRULI.NS", "ICICIGI": "ICICIGI.NS",

    # Pharmaceuticals (15)
    "SUNPHARMA": "SUNPHARMA.NS", "CIPLA": "CIPLA.NS", "DRREDDY": "DRREDDY.NS", "DIVISLAB": "DIVISLAB.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS", "MANKIND": "MANKIND.NS", "TORNTPHARM": "TORNTPHARM.NS",
    "ZYDUSLIFE": "ZYDUSLIFE.NS", "LUPIN": "LUPIN.NS", "AUROPHARMA": "AUROPHARMA.NS", "ALKEM": "ALKEM.NS",
    "BIOCON": "BIOCON.NS", "GLENMARK": "GLENMARK.NS", "LAURUSLABS": "LAURUSLABS.NS", "ABBOTINDIA": "ABBOTINDIA.NS",

    # Metals & Commodities (10)
    "GOLDBEES": "GOLDBEES.NS", "SILVERBEES": "SILVERBEES.NS", "TATASTEEL": "TATASTEEL.NS",
    "HINDALCO": "HINDALCO.NS", "VEDL": "VEDL.NS", "JSWSTEEL": "JSWSTEEL.NS", "HINDZINC": "HINDZINC.NS",
    "HDFCSILVER": "HDFCSILVER.NS", "SBISILVER": "SBISILVER.NS", "SILVERIETF": "SILVERIETF.NS",

    # Mutual Funds & Index ETFs (25+)
    "NIFTYBEES": "NIFTYBEES.NS", "BANKBEES": "BANKBEES.NS", "SETFNIF50": "SETFNIF50.NS",
    "CPSEETF": "CPSEETF.NS", "MON100": "MON100.NS", "MAFANG": "MAFANG.NS",
    "ITBEES": "ITBEES.NS", "PHARMABEES": "PHARMABEES.NS", "AUTOBEES": "AUTOBEES.NS",
    "LIQUIDBEES": "LIQUIDBEES.NS", "SETFNN50": "SETFNN50.NS",
    "AXISNIFTY": "AXISNIFTY.NS", "AXISGOLD": "AXISGOLD.NS",
    "JUNIORBEES": "JUNIORBEES.NS", "MID150BEES": "MID150BEES.NS",
    "MOMENTUM": "MOMENTUM.NS"
}

STOCK_METADATA = {
    # Banking
    "HDFCBANK": {"name": "HDFC Bank", "sector": "Banking", "ceo": "Sashidhar Jagdishan"},
    "ICICIBANK": {"name": "ICICI Bank", "sector": "Banking", "ceo": "Sandeep Bakhshi"},
    "SBIN": {"name": "State Bank of India", "sector": "Banking", "ceo": "Dinesh Kumar Khara"},
    "AXISBANK": {"name": "Axis Bank", "sector": "Banking", "ceo": "Amitabh Chaudhry"},
    "KOTAKBANK": {"name": "Kotak Mahindra Bank", "sector": "Banking", "ceo": "Ashok Vaswani"},
    "INDUSINDBK": {"name": "IndusInd Bank", "sector": "Banking", "ceo": "Sumant Kathpalia"},
    "PNB": {"name": "Punjab National Bank", "sector": "Banking", "ceo": "Atul Kumar Goel"},
    "BANKBARODA": {"name": "Bank of Baroda", "sector": "Banking", "ceo": "Debadatta Chand"},
    "FEDERALBNK": {"name": "Federal Bank", "sector": "Banking", "ceo": "Shyam Srinivasan"},
    "IDFCFIRSTB": {"name": "IDFC FIRST Bank", "sector": "Banking", "ceo": "V. Vaidyanathan"},
    "CANBK": {"name": "Canara Bank", "sector": "Banking", "ceo": "K. Satyanarayana Raju"},
    "AUBANK": {"name": "AU Small Finance Bank", "sector": "Banking", "ceo": "Sanjay Agarwal"},
    "BANDHANBNK": {"name": "Bandhan Bank", "sector": "Banking", "ceo": "Chandra Shekhar Ghosh"},
    "IDBI": {"name": "IDBI Bank", "sector": "Banking", "ceo": "Rakesh Sharma"},
    "YESBANK": {"name": "YES Bank", "sector": "Banking", "ceo": "Prashant Kumar"},
    "RBLBANK": {"name": "RBL Bank", "sector": "Banking", "ceo": "R Subramaniakumar"},

    # IT Services
    "TCS": {"name": "Tata Consultancy Services", "sector": "IT Services", "ceo": "K. Krithivasan"},
    "INFY": {"name": "Infosys", "sector": "IT Services", "ceo": "Salil Parekh"},
    "HCLTECH": {"name": "HCL Technologies", "sector": "IT Services", "ceo": "C Vijayakumar"},
    "WIPRO": {"name": "Wipro", "sector": "IT Services", "ceo": "Srini Pallia"},
    "LTIM": {"name": "LTIMindtree", "sector": "IT Services", "ceo": "Debashis Chatterjee"},
    "TECHM": {"name": "Tech Mahindra", "sector": "IT Services", "ceo": "Mohit Joshi"},
    "COFORGE": {"name": "Coforge Limited", "sector": "IT Services", "ceo": "Sudhir Singh"},
    "PERSISTENT": {"name": "Persistent Systems", "sector": "IT Services", "ceo": "Anand Deshpande"},
    "MPHASIS": {"name": "Mphasis", "sector": "IT Services", "ceo": "Nitin Rakesh"},
    "KPITTECH": {"name": "KPIT Technologies", "sector": "IT Services", "ceo": "Kishor Patil"},
    "TATAELXSI": {"name": "Tata Elxsi", "sector": "IT Services", "ceo": "Manoj Raghavan"},
    "LTTS": {"name": "L&T Technology Services", "sector": "IT Services", "ceo": "Amit Chadha"},
    "OFSS": {"name": "Oracle Financial Services", "sector": "IT Services", "ceo": "Makarand Padalkar"},
    "BSOFT": {"name": "Birlasoft", "sector": "IT Services", "ceo": "Angan Guha"},
    "CYIENT": {"name": "Cyient", "sector": "IT Services", "ceo": "Karthikeyan Natarajan"},

    # Energy
    "RELIANCE": {"name": "Reliance Industries", "sector": "Energy", "ceo": "Mukesh Ambani"},
    "ONGC": {"name": "ONGC", "sector": "Energy", "ceo": "Arun Kumar Singh"},
    "BPCL": {"name": "Bharat Petroleum", "sector": "Energy", "ceo": "G. Krishnakumar"},
    "IOC": {"name": "Indian Oil Corp", "sector": "Energy", "ceo": "Shrikant Madhav Vaidya"},
    "GAIL": {"name": "GAIL (India) Ltd", "sector": "Energy", "ceo": "Sandeep Kumar Gupta"},
    "NTPC": {"name": "NTPC Limited", "sector": "Energy", "ceo": "Gurdeep Singh"},
    "POWERGRID": {"name": "Power Grid Corp", "sector": "Energy", "ceo": "R. K. Tyagi"},
    "ADANIGREEN": {"name": "Adani Green Energy", "sector": "Energy", "ceo": "Amit Singh"},
    "ADANIENSOL": {"name": "Adani Energy Solutions", "sector": "Energy", "ceo": "Anil Sardana"},
    "TATAPOWER": {"name": "Tata Power", "sector": "Energy", "ceo": "Praveer Sinha"},
    "JSWENERGY": {"name": "JSW Energy", "sector": "Energy", "ceo": "Prashant Jain"},
    "NHPC": {"name": "NHPC Limited", "sector": "Energy", "ceo": "Rajeev Kumar Vishnoi"},
    "OIL": {"name": "Oil India Limited", "sector": "Energy", "ceo": "Ranjit Rath"},
    "PETRONET": {"name": "Petronet LNG", "sector": "Energy", "ceo": "Akshay Kumar Singh"},
    "COALINDIA": {"name": "Coal India", "sector": "Energy", "ceo": "P. M. Prasad"},

    # Automobile
    "MARUTI": {"name": "Maruti Suzuki", "sector": "Automobile", "ceo": "Hisashi Takeuchi"},
    "TATAMOTORS": {"name": "Tata Motors", "sector": "Automobile", "ceo": "Marc Llistosella"},
    "M&M": {"name": "Mahindra & Mahindra", "sector": "Automobile", "ceo": "Anish Shah"},
    "BAJAJ-AUTO": {"name": "Bajaj Auto", "sector": "Automobile", "ceo": "Rajiv Bajaj"},
    "EICHERMOT": {"name": "Eicher Motors", "sector": "Automobile", "ceo": "Siddhartha Lal"},
    "HEROMOTOCO": {"name": "Hero MotoCorp", "sector": "Automobile", "ceo": "Niranjan Gupta"},
    "TVSMOTOR": {"name": "TVS Motor Company", "sector": "Automobile", "ceo": "K. N. Radhakrishnan"},
    "ASHOKLEY": {"name": "Ashok Leyland", "sector": "Automobile", "ceo": "Shenu Agarwal"},
    "BALKRISIND": {"name": "Balkrishna Industries", "sector": "Automobile", "ceo": "Arvind Poddar"},
    "MRF": {"name": "MRF Limited", "sector": "Automobile", "ceo": "Rahul Mammen Mappillai"},
    "APOLLOTYRE": {"name": "Apollo Tyres", "sector": "Automobile", "ceo": "Neeraj Kanwar"},
    "BHARATFORG": {"name": "Bharat Forge", "sector": "Automobile", "ceo": "B. N. Kalyani"},
    "SONACOMS": {"name": "Sona BLW Precision", "sector": "Automobile", "ceo": "Vivek Vikram Singh"},
    "TIINDIA": {"name": "Tube Investments", "sector": "Automobile", "ceo": "M.A.M Arunachalam"},
    "BOSCHLTD": {"name": "Bosch Limited", "sector": "Automobile", "ceo": "Guruprasad Mudlapur"},

    # Consumer Goods (FMCG)
    "ITC": {"name": "ITC Limited", "sector": "Consumer Goods", "ceo": "Sanjiv Puri"},
    "HINDUNILVR": {"name": "Hindustan Unilever", "sector": "Consumer Goods", "ceo": "Rohit Jawa"},
    "NESTLEIND": {"name": "Nestle India", "sector": "Consumer Goods", "ceo": "Suresh Narayanan"},
    "BRITANNIA": {"name": "Britannia Industries", "sector": "Consumer Goods", "ceo": "Varun Berry"},
    "VBL": {"name": "Varun Beverages", "sector": "Consumer Goods", "ceo": "Ravi Jaipuria"},
    "TATACONSUM": {"name": "Tata Consumer Products", "sector": "Consumer Goods", "ceo": "Sunil D'Souza"},
    "GODREJCP": {"name": "Godrej Consumer", "sector": "Consumer Goods", "ceo": "Sudhir Sitapati"},
    "DABUR": {"name": "Dabur India", "sector": "Consumer Goods", "ceo": "Mohit Burman"},
    "MARICO": {"name": "Marico Limited", "sector": "Consumer Goods", "ceo": "Saugata Gupta"},
    "COLPAL": {"name": "Colgate-Palmolive", "sector": "Consumer Goods", "ceo": "Prabha Narasimhan"},
    "PGHH": {"name": "P&G Hygiene", "sector": "Consumer Goods", "ceo": "LV Vaidyanathan"},
    "BALRAMCHIN": {"name": "Balrampur Chini", "sector": "Consumer Goods", "ceo": "Vivek Saraogi"},
    "REDINGTON": {"name": "Redington Limited", "sector": "Consumer Goods", "ceo": "V. S. Hariharan"},
    "UNITDSPR": {"name": "United Spirits", "sector": "Consumer Goods", "ceo": "Hina Nagarajan"},
    "RADICO": {"name": "Radico Khaitan", "sector": "Consumer Goods", "ceo": "Lalit Khaitan"},

    # Financial Services
    "BAJFINANCE": {"name": "Bajaj Finance", "sector": "Financial Services", "ceo": "Rajeev Jain"},
    "BAJAJFINSV": {"name": "Bajaj Finserv", "sector": "Financial Services", "ceo": "Sanjiv Bajaj"},
    "CHOLAFIN": {"name": "Cholamandalam Invest", "sector": "Financial Services", "ceo": "Ravindra Kundu"},
    "SHRIRAMFIN": {"name": "Shriram Finance", "sector": "Financial Services", "ceo": "Umesh Revankar"},
    "MUTHOOTFIN": {"name": "Muthoot Finance", "sector": "Financial Services", "ceo": "George Alexander Muthoot"},
    "M&MFIN": {"name": "Mahindra & Mahindra Financial", "sector": "Financial Services", "ceo": "Raul Rebello"},
    "RECLTD": {"name": "REC Limited", "sector": "Financial Services", "ceo": "Vivek Kumar Dewangan"},
    "PFC": {"name": "Power Finance Corp", "sector": "Financial Services", "ceo": "Parminder Chopra"},
    "L&TFH": {"name": "L&T Finance Holdings", "sector": "Financial Services", "ceo": "Sudipta Roy"},
    "LICHSGFIN": {"name": "LIC Housing Finance", "sector": "Financial Services", "ceo": "Tribhuwan Adhikari"},
    "ABCAPITAL": {"name": "Aditya Birla Capital", "sector": "Financial Services", "ceo": "Vishakha Mulye"},
    "HDFCLIFE": {"name": "HDFC Life Insurance", "sector": "Financial Services", "ceo": "Vibha Padalkar"},
    "SBILIFE": {"name": "SBI Life Insurance", "sector": "Financial Services", "ceo": "Amit Jhingran"},
    "ICICIPRULI": {"name": "ICICI Pru Life", "sector": "Financial Services", "ceo": "Anup Bagchi"},
    "ICICIGI": {"name": "ICICI Lombard", "sector": "Financial Services", "ceo": "Sanjeev Mantri"},

    # Pharmaceuticals
    "SUNPHARMA": {"name": "Sun Pharmaceutical", "sector": "Pharmaceuticals", "ceo": "Dilip Shanghvi"},
    "CIPLA": {"name": "Cipla Limited", "sector": "Pharmaceuticals", "ceo": "Umang Vohra"},
    "DRREDDY": {"name": "Dr. Reddy's Lab", "sector": "Pharmaceuticals", "ceo": "G V Prasad"},
    "DIVISLAB": {"name": "Divi's Laboratories", "sector": "Pharmaceuticals", "ceo": "Kiran S. Divi"},
    "APOLLOHOSP": {"name": "Apollo Hospitals", "sector": "Pharmaceuticals", "ceo": "Prathap C. Reddy"},
    "MANKIND": {"name": "Mankind Pharma", "sector": "Pharmaceuticals", "ceo": "Rajeev Juneja"},
    "TORNTPHARM": {"name": "Torrent Pharma", "sector": "Pharmaceuticals", "ceo": "Samir Mehta"},
    "ZYDUSLIFE": {"name": "Zydus Lifesciences", "sector": "Pharmaceuticals", "ceo": "Sharvil Patel"},
    "LUPIN": {"name": "Lupin Limited", "sector": "Pharmaceuticals", "ceo": "Nilesh Gupta"},
    "AUROPHARMA": {"name": "Aurobindo Pharma", "sector": "Pharmaceuticals", "ceo": "K. Nithyananda Reddy"},
    "ALKEM": {"name": "Alkem Laboratories", "sector": "Pharmaceuticals", "ceo": "Sandeep Singh"},
    "BIOCON": {"name": "Biocon Limited", "sector": "Pharmaceuticals", "ceo": "Kiran Mazumdar-Shaw"},
    "GLENMARK": {"name": "Glenmark Pharma", "sector": "Pharmaceuticals", "ceo": "Glenn Saldanha"},
    "LAURUSLABS": {"name": "Laurus Labs", "sector": "Pharmaceuticals", "ceo": "Satyanarayana Chava"},
    "ABBOTINDIA": {"name": "Abbott India", "sector": "Pharmaceuticals", "ceo": "Vivek V. Kamath"},

    # Metals & Commodities
    "GOLDBEES": {"name": "Gold BEES ETF", "sector": "Commodities", "ceo": "N/A"},
    "SILVERBEES": {"name": "Silver BEES ETF", "sector": "Mutual Funds", "cap": "Commodities", "ceo": "N/A"},
    "TATASTEEL": {"name": "Tata Steel", "sector": "Metals", "ceo": "T. V. Narendran"},
    "HINDALCO": {"name": "Hindalco Industries", "sector": "Metals", "ceo": "Satish Pai"},
    "VEDL": {"name": "Vedanta Limited", "sector": "Metals", "ceo": "Sunil Duggal"},
    "JSWSTEEL": {"name": "JSW Steel", "sector": "Metals", "ceo": "Sajjan Jindal"},
    "HINDZINC": {"name": "Hindustan Zinc", "sector": "Metals", "ceo": "Arun Misra"},
    "HDFCSILVER": {"name": "HDFC Silver ETF", "sector": "Mutual Funds", "cap": "Commodities", "ceo": "N/A"},
    "SBISILVER": {"name": "SBI Silver ETF", "sector": "Mutual Funds", "cap": "Commodities", "ceo": "N/A"},
    "SILVERIETF": {"name": "ICICI Prudential Silver ETF", "sector": "Mutual Funds", "cap": "Commodities", "ceo": "N/A"},

    # Mutual Funds
    "NIFTYBEES": {"name": "Nippon India Nifty 50 ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "BANKBEES": {"name": "Nippon India Bank ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "SETFNIF50": {"name": "SBI Nifty 50 ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "CPSEETF": {"name": "CPSE ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "MON100": {"name": "Motilal Oswal Nasdaq 100 ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "MAFANG": {"name": "Mirae Asset NYSE FANG+ ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "ITBEES": {"name": "Nippon India ETF Nifty IT", "sector": "Mutual Funds", "cap": "Sectoral", "ceo": "N/A"},
    "PHARMABEES": {"name": "Nippon India ETF Nifty Pharma", "sector": "Mutual Funds", "cap": "Sectoral", "ceo": "N/A"},
    "AUTOBEES": {"name": "Nippon India ETF Nifty Auto", "sector": "Mutual Funds", "cap": "Sectoral", "ceo": "N/A"},
    "LIQUIDBEES": {"name": "Nippon India ETF Liquid BeES", "sector": "Mutual Funds", "cap": "Debt", "ceo": "N/A"},
    "SETFNN50": {"name": "SBI ETF Nifty Next 50", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "AXISNIFTY": {"name": "Axis Nifty 50 ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "AXISGOLD": {"name": "Axis Gold ETF", "sector": "Mutual Funds", "cap": "Commodities", "ceo": "N/A"},
    "JUNIORBEES": {"name": "Nippon India Nifty Next 50 ETF", "sector": "Mutual Funds", "cap": "Large Cap", "ceo": "N/A"},
    "MID150BEES": {"name": "Nippon India Nifty Midcap 150 ETF", "sector": "Mutual Funds", "cap": "Mid Cap", "ceo": "N/A"},
    "MOMENTUM": {"name": "UTI Nifty200 Momentum 30 ETF", "sector": "Mutual Funds", "cap": "Mid Cap", "ceo": "N/A"}
}

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- Static File Serving (Last) ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
async def read_index():
    index_path = os.path.join(root_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "index.html not found in root", "root": root_dir}

# Mount static files at /static for backward compatibility and explicit access
app.mount("/static", StaticFiles(directory=root_dir, html=True), name="static_dir")

# Mount root last as a fallback for CSS/JS/Images
app.mount("/", StaticFiles(directory=root_dir, html=True), name="root_static")

if __name__ == "__main__":
    import uvicorn
    # Important for deployment: bind to PORT env var or 8001
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
