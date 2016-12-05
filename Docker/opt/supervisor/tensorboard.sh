#!/bin/bash

function shutdown()
{
    echo "`date +"%d.%m.%Y %T.%3N"` - Shutting down tensorboard"
}

echo "`date +"%d.%m.%Y %T.%3N"` - Starting tensorboard"
tensorboard --logdir /content/logs/


# Allow any signal which would kill a process to stop server
trap shutdown HUP INT QUIT ABRT KILL ALRM TERM TSTP


