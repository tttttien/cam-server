#!/bin/bash

# ✅ Set ngrok authtoken if provided (only needed first time or if using env variable)
if [ -n "$NGROK_AUTHTOKEN" ]; then
  echo "[+] Setting ngrok authtoken..."
  ngrok config add-authtoken "$NGROK_AUTHTOKEN"
fi

# ✅ Start ngrok in the background forwarding port 8888
echo "[+] Starting ngrok..."
ngrok http 8888 --log=stdout > /tmp/ngrok.log &

# ✅ Wait for ngrok to be ready (up to 10 seconds)
for i in {1..10}; do
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o 'https://[0-9a-z]*\.ngrok.io')
  if [[ -n "$NGROK_URL" ]]; then
    echo "[+] Ngrok URL: $NGROK_URL"
    break
  fi
  sleep 1
done

# ✅ Start FastAPI server
echo "[+] Starting FastAPI server..."
uvicorn main:app --host 0.0.0.0 --port 8888
