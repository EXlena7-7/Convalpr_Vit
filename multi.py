import socket, cv2, pickle, struct
import imutils
import threading
import cv2

server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ip = 'rtsp://admin:Vt3lc4123@38.51.120.236:8061'
port = '8061'
socket_address = (host_ip,port)
server_socket.bind(socket_address)
server_socket.listen()
print("Listening at",socket_address)

def start_video_stream():
    client_socket,addr=server_socket.accept()
    camera=False
    if camera ==True:
        vid= cv2.VideoCapture(0)
    else:
        vid=cv2.VideoCapture('./assets/sample.mp4')
    try:
        print('CLIENT {} CONNECTED!'.format(addr))
        if client_socket:
            while(vid.isOpened()):
                img,frame = vid.read()
                
                frame = imutils.resize(frame,width=320)
                a = pickle.dumps(frame)
                message =struct.pack("Q",len(a))+a
                client_socket.sendall(message)
                cv2.imshow("TRANSMITING TO CACHE SERVER", frame)
                key= cv2.waitKey(1) & 0xFF
                if key==ord('q'):
                    client_socket.close()
                    break
    except Exception as e:
        print(f'CACHE SERVER {addr} DISCONNECTED')
        pass
while True:
    start_video_stream()