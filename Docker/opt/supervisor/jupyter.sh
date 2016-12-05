#!/bin/bash

function shutdown()
{
    echo "`date +"%d.%m.%Y %T.%3N"` - Shutting down jupyter"
}

echo "`date +"%d.%m.%Y %T.%3N"` - Starting jupyter"
jupyter notebook "$@"


# Allow any signal which would kill a process to stop server
trap shutdown HUP INT QUIT ABRT KILL ALRM TERM TSTP

