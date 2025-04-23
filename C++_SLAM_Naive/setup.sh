docker run \
    --rm \
    --mount type=bind, source=$(pwd),target=/slam \
    --network=host \
    --env DISPLAY=$(DISPLAY) \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --device=/dev/dri:/dev/dri \
    --device=/dev/video0:/dev/video0 \
    --interactive \
    --tty \
    $(docker build -q .)