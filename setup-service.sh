#!/bin/bash
set -e

# Kill any existing nohup instance
pkill -f "scripts/serve.py" 2>/dev/null || true
sleep 2

# Create systemd service
tee /etc/systemd/system/epstein-rag.service > /dev/null << 'EOF'
[Unit]
Description=Epstein Documents RAG Server
After=network.target

[Service]
Type=simple
User=austin
WorkingDirectory=/storage/epstein_llm
ExecStart=/storage/epstein_llm/venv/bin/python scripts/serve.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
systemctl daemon-reload
systemctl enable epstein-rag
systemctl start epstein-rag

echo "Done. Checking status..."
sleep 3
systemctl status epstein-rag --no-pager
