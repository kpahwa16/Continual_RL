import socket, pickle, time
import pdb, sys


def server_connect(PORT=3333,
                   HOST="0.0.0.0",
                   num_listeners=4):
    print("Server is Listening.....")
    sock = server_init(PORT=PORT,
                       HOST=HOST, 
                       num_listeners=num_listeners)
    connection, _ = sock.accept()
    return connection, sock
    # return sock

def server_init(PORT=3333,
                HOST="0.0.0.0",
                num_listeners=4):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((HOST, PORT))
    sock.listen(num_listeners)
    return sock


def server_disconnect(connection, sock):
    print("Server shutdown")
    connection.close()
    sock.shutdown(1)
    sock.close()

def send_seq(connection, sequence):
    byte_sequence = pickle.dumps(sequence)
    eos_token = bytes("$$$$$_$$$$$", "utf-8")
    connection.send(byte_sequence)
    connection.send(eos_token)

def receive_seq(sock, num_bytes=8192):
    data = None
    whole_data = b""
    while True:
        data = sock.recv(num_bytes)
        whole_data = whole_data + data
        if b'$$$$$_$$$$$' in whole_data:
            break
    time.sleep(0.01)
    # print("Line 47: pickle.loads(whole_data[:-11]) = {}".format(pickle.loads(whole_data[:-11])))
    return pickle.loads(whole_data[:-11])

def send_simp(connection, string):
    data = bytes(string, "utf-8")
    connection.send(data)

def receive_simp(s, num_bytes=1024):
    data = s.recv(num_bytes)
    time.sleep(0.01)
    return data.decode()

def server_send(s,
                connection,
                send_file_path="membuf_Krull-ram-v0_agent-1-task-3.pkl",
                num_bytes=8192):
    ONE_CONNECTION_ONLY = True
    start_time = time.time()
    total_len = 0
    while True:
        with open(send_file_path, "rb") as file:
            data = file.read(num_bytes)

            # print(data.decode())
            # sys.stdout.flush()
            while True:
                total_len += len(data)
                # print("send:", total_len)
                if not data:
                    # pdb.set_trace()
                    data = bytes("$$$$$_$$$$$", "utf-8")
                    connection.send(data)
                    break
                connection.send(data)
                data = file.read(num_bytes)
        if ONE_CONNECTION_ONLY:
            break
    end_time = time.time()
    print("Duration of server_send(): {}".format(end_time - start_time))
    print("Complete sending the file")

def server_send_simp(connection, string):
    data = bytes(string, "utf-8")
    connection.send(data)

def client_connect(PORT=1219, HOST="10.161.159.156"):
    # the current setting runs Atari server on Xavier, and agent 2 (w/o send_first) on Xavier, 
    #     and agent 1 with send_first on Nano
    # when connecting to remote server running on the same device (Xavier), use 10.161.159.142, 
    #     when connecting to the other agent, use "10.161.159.156" (Nano)

    # HOST = "10.161.159.142" # "10.161.159.156" # socket.gethostname()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # pdb.set_trace()
    sock.connect((HOST, PORT))
    return sock

def client_disconnect(sock):
    sock.close()
    print("Connection is closed")

def client_receive_simp(s, num_bytes=1024):
    data = s.recv(num_bytes)
    time.sleep(0.01)
    return data.decode()

def client_receive(s,
                   recv_file_path="target.pkl",
                   num_bytes=8192):
    start_time = time.time()
    data = None
    whole_data = b""
    total_len = 0
    with open(recv_file_path, "wb") as file:
        print("File open")
        print("Receiving data...")
        while True:
            data = s.recv(num_bytes)
            total_len += len(data)
            # print("receive:", total_len)
            
            whole_data = whole_data + data
            
            # print("recv data:", data, whole_data)
            
            if b'$$$$$_$$$$$' in whole_data:
                # pdb.set_trace() 
                file.write(whole_data[:-11])
                # pdb.set_trace()
                break     
    end_time = time.time()
    print("Duration of client_receive(): {}".format(end_time - start_time))
    print("Got the file")
    

def send_server(PORT=1218, 
                num_listeners=2,
                send_file_path="membuf_Krull-ram-v0_agent-1-task-3.pkl",
                num_bytes=1024):
    print("Server is Listening.....")
    HOST = socket.gethostname() # 'localhost'
    # PORT = 1218 # 50007
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(num_listeners)

    # filename = "membuf_Krull-ram-v0_agent-1-task-3.pkl" # "dummy.txt"
    ONE_CONNECTION_ONLY = True
    print("File server started...")
    start_time = time.time()
    while True:
        connection, address = s.accept()
        print("Accepted connection from {}".format(address))
        with open(send_file_path, "rb") as file:
            data = file.read(num_bytes)
            # pdb.set_trace()
            while data:
                connection.send(data)
                data = file.read(num_bytes)
        print("File sent complete")
        connection.close()
        if ONE_CONNECTION_ONLY:
            break
    s.shutdown(1)
    s.close()
    end_time = time.time()
    print("Time Duration: {}".format(end_time - start_time))
    print('Data received from client')


def receive_client(PORT=1218, 
           recv_file_path="target.pkl",
           num_bytes=1024):
    HOST = socket.gethostname() # 'localhost'
    # PORT = 1218 # 50007
    # Create a socket connection.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    start_time = time.time()
    # filename = "target.pkl"

    data = None
    with open(recv_file_path, "wb") as file:
        print("File open")
        print("Receiving data...")
        while True:
            data = s.recv(num_bytes)
            if not data:
                break
            file.write(data)
            # file.loads(data)
    print("Got the file")
    end_time = time.time()
    print("Time Duration: {}".format(end_time - start_time))
    s.close()
    print("Connection is closed")


def wait_execution(duration=15):
    print("waiting...")
    time.sleep(duration)
