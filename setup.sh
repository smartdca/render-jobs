#!/bin/bash
# put the render code and assets at the same absolute paths they use on the workstation
set -e
sudo apt-get install -y -qq ffmpeg > /dev/null
python3 -m pip install -q numpy==2.5.3 opencv-python-headless==5.0.0.93 pillow==12.3.0 scipy || \
python3 -m pip install -q numpy opencv-python-headless pillow scipy
sudo mkdir -p /home/claude && sudo chown -R $(id -u):$(id -g) /home/claude
cp -r root/home/claude/. /home/claude/
sudo mkdir -p /usr/share/fonts/opentype/noto /usr/share/fonts/truetype/dejavu
sudo cp fonts/NotoSansCJK-*.ttc /usr/share/fonts/opentype/noto/; sudo cp fonts/DejaVu*.ttf /usr/share/fonts/truetype/dejavu/
python3 -c "import numpy, cv2, PIL; print('libs', numpy.__version__, cv2.__version__, PIL.__version__)"
