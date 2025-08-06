#!/bin/bash

echo "Re-executing systemctl..."
sudo systemctl daemon-reexec

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Restarting uvicorn service..."
sudo systemctl restart uvicorn

echo "Done. Checking logs..."
sudo journalctl -u uvicorn -f
