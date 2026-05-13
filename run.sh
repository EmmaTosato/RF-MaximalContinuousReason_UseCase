#!/usr/bin/env bash
# ==========================================
# MCR - Docker Management
# ==========================================

set -e

IMAGE_NAME="mcr:latest"
CONTAINER_NAME="mcr-container"
REDIS_PORT=6379
JUPYTER_PORT=8888

# Parse named options before the command
while [[ "$1" == --* ]]; do
    case "$1" in
        --redis-port)   REDIS_PORT="$2";   shift 2 ;;
        --jupyter-port) JUPYTER_PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; echo "Run './run.sh help' for usage."; exit 1 ;;
    esac
done

CMD="${1:-start}"

case "$CMD" in
    start)
        echo ""
        echo "=========================================="
        echo "  MCR - Starting Container"
        echo "=========================================="
        echo ""

        if ! command -v docker &> /dev/null; then
            echo "[ERROR] Docker not found!"
            echo "Install Docker: https://docs.docker.com/get-docker/"
            exit 1
        fi
        echo "[1/4] Checking Docker... OK"
        echo ""

        echo "[2/4] Building image $IMAGE_NAME..."
        docker build -t "$IMAGE_NAME" .
        echo "Build completed!"
        echo ""

        echo "[3/4] Removing existing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        echo ""

        echo "[4/4] Starting container $CONTAINER_NAME..."
        echo "      Redis port:   $REDIS_PORT"
        echo "      Jupyter port: $JUPYTER_PORT"
        docker run -d \
            --name "$CONTAINER_NAME" \
            -p "${REDIS_PORT}":6379 \
            -p "${JUPYTER_PORT}":8888 \
            -v "$(pwd)/logs:/app/logs" \
            -v "$(pwd)/workers:/app/workers" \
            -v "$(pwd)/results:/app/results" \
            -v "$(pwd)/fig:/app/fig" \
            "$IMAGE_NAME" || {
                echo ""
                echo "[ERROR] Failed to start container."
                echo "        If the error is 'port is already allocated', use different ports:"
                echo "        ./run.sh start --redis-port 7379 --jupyter-port 9888"
                exit 1
            }

        echo ""
        echo "=========================================="
        echo "  Container started successfully!"
        echo "=========================================="
        echo ""
        echo "Container: $CONTAINER_NAME"
        echo "Redis:     localhost:${REDIS_PORT}"
        echo "Jupyter:   http://localhost:${JUPYTER_PORT}"
        echo ""
        echo "Commands:"
        echo "  ./run.sh stop      Stop container"
        echo "  ./run.sh shell     Open shell"
        echo "  ./run.sh logs      View logs"
        echo ""
        ;;

    stop)
        echo ""
        echo "Stopping container $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME"
        echo "Container stopped."
        echo ""
        ;;

    restart)
        echo ""
        echo "Restarting container $CONTAINER_NAME..."
        docker restart "$CONTAINER_NAME"
        echo "Container restarted."
        echo ""
        ;;

    shell)
        echo ""
        echo "Opening shell in $CONTAINER_NAME..."
        docker exec -it "$CONTAINER_NAME" bash
        ;;

    logs)
        docker logs -f "$CONTAINER_NAME"
        ;;

    rebuild)
        echo ""
        echo "=========================================="
        echo "  MCR - Rebuilding Image"
        echo "=========================================="
        echo ""
        docker build --no-cache -t "$IMAGE_NAME" .
        echo "Build completed!"
        echo ""
        "$0" --redis-port "$REDIS_PORT" --jupyter-port "$JUPYTER_PORT" start
        ;;

    clean-rebuild)
        echo ""
        echo "=========================================="
        echo "  MCR - Clean Rebuild"
        echo "=========================================="
        echo ""
        echo "Removing container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
        "$0" --redis-port "$REDIS_PORT" --jupyter-port "$JUPYTER_PORT" rebuild
        ;;

    help|--help|-h)
        echo ""
        echo "Usage: ./run.sh [--redis-port PORT] [--jupyter-port PORT] [command]"
        echo ""
        echo "Commands:"
        echo "  start         Build and start container (default)"
        echo "  stop          Stop container"
        echo "  restart       Restart container"
        echo "  shell         Open bash shell in container"
        echo "  logs          Show container logs"
        echo "  rebuild       Rebuild image (no cache)"
        echo "  clean-rebuild Remove container and rebuild (no cache)"
        echo "  help          Show this help"
        echo ""
        echo "Options:"
        echo "  --redis-port PORT    Host port for Redis   (default: 6379)"
        echo "  --jupyter-port PORT  Host port for Jupyter (default: 8888)"
        echo ""
        echo "Examples:"
        echo "  ./run.sh"
        echo "  ./run.sh start --redis-port 7379 --jupyter-port 9888"
        echo ""
        ;;

    *)
        echo "Unknown command: $CMD"
        echo "Run './run.sh help' for usage."
        exit 1
        ;;
esac
