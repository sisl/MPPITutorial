import pickle
import numpy as np
import socket as Socket
from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()
TCP_PORTS = None
TCP_SIZE = 0
TCP_RANK = 0

def set_rank_size(tcp_rank, tcp_ports):
	global TCP_PORTS, TCP_SIZE, TCP_RANK
	TCP_PORTS, TCP_SIZE, TCP_RANK = tcp_ports, len(tcp_ports), tcp_rank
	assert min(MPI_RANK, TCP_RANK) == 0, "Maximum one of the TCP or MPI ranks can be greater than zero"
	rank = max(MPI_RANK, TCP_RANK)
	size = max(len(TCP_PORTS), MPI_SIZE)
	return rank, size

def get_rank_size():
	return max(MPI_RANK, TCP_RANK), max(TCP_SIZE, MPI_SIZE)

def get_client(server_ports):
	return TCPClient(server_ports) if MPI_SIZE==1 else MPIConnection(cluster=server_ports)

def get_server(root=0):
	return TCPServer() if MPI_SIZE==1 else MPIConnection(root=root)

class TCPServer():
	def __init__(self):
		self.port = TCP_PORTS[TCP_RANK]
		self.sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
		self.sock.setsockopt(Socket.SOL_SOCKET, Socket.SO_REUSEADDR, 1)
		self.sock.bind(("localhost", self.port))
		self.sock.listen(5)
		print(f"Worker listening on port {self.port} ...")
		self.conn = self.sock.accept()[0]
		print(f"Connected!")

	def recv(self, decoder=pickle.loads):
		return decoder(self.conn.recv(1000000))

	def send(self, data, encoder=pickle.dumps):
		self.conn.sendall(encoder(data))

	def __del__(self):
		self.conn.close()
		self.sock.close()

class TCPClient():
	def __init__(self, client_ranks):
		self.num_clients = len(client_ranks)
		self.client_ranks = sorted(client_ranks)
		self.client_ports = [TCP_PORTS[rank] for rank in self.client_ranks]
		self.client_sockets = self.connect_sockets(self.client_ports)

	def connect_sockets(self, ports):
		client_sockets = {port:None for port in ports}
		for port in ports:
			sock = Socket.socket(Socket.AF_INET, Socket.SOCK_STREAM)
			sock.connect(("localhost", port))
			client_sockets[port] = sock
		return client_sockets

	def broadcast(self, params, encoder=pickle.dumps):
		num = min(len(params), self.num_clients)
		[self.send(encoder(p), rank) for p,rank in zip(params[:num], self.client_ranks[:num])]
			
	def send(self, param, rank):
		self.client_sockets[TCP_PORTS[rank]].sendall(param)

	def gather(self, decoder=pickle.loads):
		return [decoder(sock.recv(1000000)) for port, sock in self.client_sockets.items()]

	def __del__(self):
		for sock in self.client_sockets.values(): sock.close()

class MPIConnection():
	def __init__(self, cluster=None, root=0):
		self.root = root
		self.rank = MPI_RANK
		self.cluster = sorted(cluster) if isinstance(cluster, list) else []

	def broadcast(self, params, encoder=lambda x: x):
		[MPI_COMM.send(encoder(p), dest=i, tag=self.rank) for p,i in zip(params, self.cluster)]
			
	def gather(self, decoder=lambda x: x):
		return [decoder(MPI_COMM.recv(source=i, tag=self.rank)) for i in self.cluster]

	def send(self, data, encoder=lambda x: x):
		MPI_COMM.send(encoder(data), dest=self.root, tag=self.root)

	def recv(self, decoder=lambda x: x):
		param = decoder(MPI_COMM.recv(source=self.root, tag=self.root))
		return param
