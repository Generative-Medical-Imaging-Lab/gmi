# GMI Development Guide

## Overview

This project uses a Docker container for a controlled development environment. The container is already running and contains all necessary dependencies. We develop the GMI package by editing files on the host system and testing them inside the container.

## Key Concepts

- **Host System**: Your local machine where you edit files
- **Docker Container**: Running environment with all dependencies
- **No Rebuilding**: The container is already set up and running
- **Live Development**: Edit files on host, test immediately in container

## Container Management

### Container Status
The container is already running and ready to use. You don't need to rebuild or restart it.

### Accessing the Container
```bash
# Execute commands inside the running container
docker exec -it gmi-container <command>

# Examples:
docker exec -it gmi-container python main.py --help
docker exec -it gmi-container python examples/modular_configs/run_default_study.py
docker exec -it gmi-container bash examples/modular_configs/run_all_studies_cli.sh
```

### Why Docker Exec?
- `docker exec` runs commands inside the existing container
- `-it` provides interactive terminal with proper formatting
- Changes to files on the host are immediately available in the container
- No need to rebuild or restart the container

## Development Workflow

### 1. Edit Files
Edit GMI package files on your host system:
- `gmi/` - Main package code
- `examples/` - Example scripts and configs
- `main.py` - CLI entry point

### 2. Test Changes
Run tests inside the container to verify changes:
```bash
# Test a single component
docker exec -it gmi-container python -c "from gmi.datasets.mnist import MNIST; print('MNIST works!')"

# Test CLI command
docker exec -it gmi-container python main.py train-image-reconstructor examples/modular_configs/training_config.yaml

# Test example script
docker exec -it gmi-container python examples/modular_configs/run_default_study.py
```

### 3. Debug Issues
When you encounter errors:

1. **Read the Stack Trace**: Look for file paths and line numbers
2. **Identify the Problem**: Find the specific function/class causing issues
3. **Edit the File**: Make changes on the host system
4. **Test Again**: Run the same command to verify the fix
5. **Iterate**: Repeat until the issue is resolved

### 4. Run Examples
Once components are working, run full examples:
```bash
# Run modular studies
docker exec -it gmi-container bash examples/modular_configs/run_all_studies_cli.sh

# Run individual studies
docker exec -it gmi-container python examples/modular_configs/run_all_modular_studies.py
```

## Common Commands

### Package Development
```bash
# Test import
docker exec -it gmi-container python -c "import gmi; print('Package loaded successfully')"

# Test specific module
docker exec -it gmi-container python -c "from gmi.commands.train_image_reconstructor import train_image_reconstructor_from_configs"

# Test dataset
docker exec -it gmi-container python -c "from gmi.datasets.mnist import MNIST; dataset = MNIST(); print(f'Dataset size: {len(dataset)}')"
```

### CLI Testing
```bash
# Test CLI help
docker exec -it gmi-container python main.py --help

# Test specific command
docker exec -it gmi-container python main.py train-image-reconstructor examples/modular_configs/training_config.yaml --experiment-name test

# Test with overrides
docker exec -it gmi-container python main.py train-image-reconstructor examples/modular_configs/training_config.yaml --train-dataset examples/modular_configs/datasets/bloodmnist_train.yaml --image-reconstructor examples/modular_configs/image_reconstructors/linear_conv_3ch.yaml
```

### Example Scripts
```bash
# Run default study
docker exec -it gmi-container python examples/modular_configs/run_default_study.py

# Run all modular studies (Python)
docker exec -it gmi-container python examples/modular_configs/run_all_modular_studies.py

# Run all modular studies (Bash)
docker exec -it gmi-container bash examples/modular_configs/run_all_studies_cli.sh
```

## Debugging Tips

### 1. Check Container Status
```bash
# Verify container is running
docker ps

# Check container logs
docker logs gmi-container
```

### 2. Interactive Debugging
```bash
# Start interactive Python session
docker exec -it gmi-container python

# Or start bash session
docker exec -it gmi-container bash
```

### 3. File System
- Files edited on host are immediately available in container
- Container path: `/gmi_base/` (maps to your project root)
- Working directory in container: `/gmi_base/`

### 4. Common Issues
- **Import Errors**: Check if module exists and is properly imported
- **Path Issues**: Use relative paths from `/gmi_base/` in container
- **Permission Issues**: Files should be readable by container user
- **Memory Issues**: Large datasets may require more container memory

## Project Structure

```
gmi/                          # Main package
├── commands/                 # CLI commands
├── datasets/                 # Dataset implementations
├── network/                  # Neural network architectures
├── tasks/                    # Training tasks
└── ...

examples/                     # Example scripts and configs
├── modular_configs/         # Modular configuration system
│   ├── datasets/            # Dataset configs
│   ├── measurement_simulators/  # Noise configs
│   ├── image_reconstructors/    # Network configs
│   └── run_*.py            # Example scripts
└── ...

main.py                      # CLI entry point
requirements.txt             # Python dependencies
Dockerfile                   # Container definition
docker-compose.yml          # Container orchestration
```

## Best Practices

1. **Test Incrementally**: Test small changes before running full examples
2. **Use Stack Traces**: Always read error messages carefully
3. **Check Imports**: Verify all imports work before running scripts
4. **Monitor Resources**: Watch for memory/CPU usage in long-running tasks
5. **Save Configs**: The system automatically saves final configs to experiment directories

## Troubleshooting

### Container Not Responding
```bash
# Restart container
docker restart gmi-container

# Check container status
docker ps -a
```

### Permission Issues
```bash
# Check file permissions
docker exec -it gmi-container ls -la /gmi_base/

# Fix permissions if needed (run on host)
chmod -R 755 .
```

### Memory Issues
```bash
# Check container resource usage
docker stats gmi-container

# Increase memory limit if needed (edit docker-compose.yml)
```

This development environment provides a controlled, reproducible setup for developing and testing the GMI package. The key is to understand that you're editing on the host but testing in the container, and the `docker exec` command is your bridge between the two. 