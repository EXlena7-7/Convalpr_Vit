#!/bin/bash

exec python3 --version | python --version

NAME=pruebas

DIR=$PWD

SRC=$DIR/env/bin/activate

ENV="env"

cd $DIR

cd app

ls

exec uvicorn main:app --host 0.0.0.0 --port  8888 --reload
