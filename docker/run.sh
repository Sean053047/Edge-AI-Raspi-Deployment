#! /bin/bash

docker run -it --rm \
    --name edge-ai \
    -v $PWD:/workspace \
    -v $HOME/.ssh/authorized_keys:/home/itri/.ssh/authorized_keys \
    -v /dev:/dev \
    -v /lib/firmware:/lib/firmware \
    -v /lib/udev/rules.d:/lib/udev/rules.d \
    -v /lib/modules:/lib/modules \
    --device=/dev/hailo0:/dev/hailo0 \
    --net host \
    edge-ai \
    bash -c "sudo apt-get install -f && sudo service ssh start && bash"
