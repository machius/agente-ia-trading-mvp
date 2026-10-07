import yfinance as yf
import json
print(len(yf.Ticker("AAPL").news))
print(len(yf.Ticker("^GSPC").news))
print(len(yf.Ticker("GC=F").news))
print(len(yf.Ticker("BTC-USD").news))