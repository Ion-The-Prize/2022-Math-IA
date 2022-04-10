FROM phusion/baseimage:focal-1.2.0

RUN apt-get update 
# headless x11 that you can connect to with web
RUN apt-get install -y x11vnc xvfb xterm novnc ctwm
# netstat needed by novnc
RUN apt-get install -y net-tools 

# Python basics
RUN apt-get install -y python3 python3-pip

# Hoping to use this to add pycharm
RUN apt-get install -y snapd

# Python and OS packages needed for Math-IA/Polynomial work
RUN pip3 install numpy styleframe matplotlib mplcursors webcolors sympy pandas
RUN apt-get install -y python3-tk python3-pil.imagetk

RUN apt-get install -y wm2
ENV WIN_MGR=wm2

ENV DISPLAY=:20
WORKDIR /project

RUN mkdir /etc/service/xvfb
RUN echo '#!/bin/bash\n exec > >(sed 's/^/xvfb:/') 2>&1; /usr/bin/Xvfb $DISPLAY -screen 0 1024x768x16 -cc 4 -listen tcp' > /etc/service/xvfb/run
RUN chmod +x /etc/service/xvfb/run

RUN mkdir /etc/service/winmgr
RUN echo "#!/bin/bash\n exec > >(sed 's/^/xtwm:/') 2>&1; sv status /etc/service/xvfb || exit 1; $WIN_MGR" > /etc/service/winmgr/run
RUN chmod +x /etc/service/winmgr/run

RUN mkdir /etc/service/xterm
RUN echo "#!/bin/bash\n exec > >(sed 's/^/xterm:/') 2>&1; sv status /etc/service/xvfb || exit 1; cd /project; xterm" > /etc/service/xterm/run
RUN chmod +x /etc/service/xterm/run

RUN mkdir /etc/service/x11vnc
RUN echo "#!/bin/bash\n exec > >(sed 's/^/x11vnc:/') 2>&1; sv status /etc/service/xvfb || exit 1; x11vnc -nopw -ncache 10 -ncache_cr -shared -forever" > /etc/service/x11vnc/run
RUN chmod +x /etc/service/x11vnc/run

RUN mkdir /etc/service/novnc
RUN echo "#!/bin/bash\n exec > >(sed 's/^/novnc:/') 2>&1; sv status /etc/service/x11vnc || exit 1; /usr/share/novnc/utils/launch.sh" > /etc/service/novnc/run
RUN chmod +x /etc/service/novnc/run

RUN mkdir -p /project
COPY *.py /project

RUN echo "exec xterm" > /root/.xinitrc && chmod +x /root/.xinitrc
#CMD bash

