#!/bin/bash

set -e


echo "=>Starting Supervisor."
# echo "You can safely CTRL-C and the container will continue to run..."
/usr/bin/supervisord -c /etc/supervisor/conf.d/tensorboard_jupyter.conf >> /dev/null
