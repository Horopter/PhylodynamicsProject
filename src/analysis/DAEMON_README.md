# Daemon Runner for Analyses

This daemon process automatically runs all analyses (MLE, Bayesian, PhyloDeep) in parallel, waits for completion, and then runs the comparison.

## Features

- **Background Execution**: Runs as a daemon process, independent of terminal
- **Lid-Close Resistant**: Continues running even when laptop lid is closed
- **Parallel Execution**: Runs all three analyses simultaneously
- **Automatic Comparison**: Triggers comparison automatically when analyses complete
- **Comprehensive Logging**: Logs all output to files for monitoring
- **Process Management**: Easy start/stop/status commands

## Usage

### Start Daemon

```bash
# Start the daemon in background
python src/analysis/daemon_runner.py start
```

The daemon will:
1. Start all three analyses (MLE, Bayesian, PhyloDeep) in parallel
2. Monitor their progress
3. Automatically run comparison when analyses complete
4. Log everything to `output/daemon/daemon.log`

### Check Status

```bash
# Check if daemon is running
python src/analysis/daemon_runner.py status
```

### Stop Daemon

```bash
# Stop the running daemon
python src/analysis/daemon_runner.py stop
```

### Restart Daemon

```bash
# Restart the daemon
python src/analysis/daemon_runner.py restart
```

## Monitoring

### View Logs

```bash
# View daemon log (real-time)
tail -f output/daemon/daemon.log

# View stdout
tail -f output/daemon/daemon_stdout.log

# View stderr
tail -f output/daemon/daemon_stderr.log
```

### Check Process

```bash
# Find daemon process
ps aux | grep daemon_runner

# Check resource usage
python src/analysis/daemon_runner.py status
```

## How It Works

1. **Daemon Process**: Uses Unix double-fork technique to become a proper daemon
2. **Signal Handling**: Responds to SIGTERM/SIGINT for graceful shutdown
3. **PID Management**: Tracks daemon PID in `output/daemon/daemon.pid`
4. **Parallel Execution**: Uses ProcessPoolExecutor to run analyses simultaneously
5. **Automatic Comparison**: Monitors analysis completion and triggers comparison

## Persistence

The daemon is designed to survive:
- Terminal closure
- SSH session disconnection
- Laptop lid closing (on most systems)
- Network interruptions

## Troubleshooting

### Daemon won't start
- Check if another daemon is already running: `python daemon_runner.py status`
- Check logs: `tail -f output/daemon/daemon.log`
- Remove stale PID file: `rm output/daemon/daemon.pid`

### Daemon stops unexpectedly
- Check system logs: `dmesg | tail` or `journalctl -xe`
- Check daemon logs: `tail -f output/daemon/daemon_stderr.log`
- Verify dependencies are installed: `pip install psutil`

### Can't stop daemon
- Force kill: `kill -9 $(cat output/daemon/daemon.pid)`
- Or: `pkill -f daemon_runner.py`

## Notes

- The daemon runs analyses with default settings (all CPUs, default sampling probability)
- Bayesian analysis is optional - daemon continues even if it fails
- Comparison only runs if at least one analysis completes successfully
- All output is logged for later review

