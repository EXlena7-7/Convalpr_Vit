FROM tensorflow/tensorflow:2.15.0-gpu

RUN python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices())"

# Install wget
RUN apt-get update && apt-get install -y wget dos2unix

WORKDIR /app

COPY requirements.txt .

ENV TZ=America/Caracas

RUN ln -sf /usr/share/zoneinfo/America/Caracas /etc/localtime && \
    echo "America/Caracas" > /etc/timezone

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libpng16-16


RUN python --version

RUN pip install -r requirements.txt

RUN pip install "uvicorn[standard]"

RUN pip install fastapi

RUN ls

RUN pip list


# Convert line endings of uvicorn_start.sh to Unix style
#RUN dos2unix uvicorn_start.sh


EXPOSE 8888

CMD ["bash", "./uvicorn_start.sh"]
