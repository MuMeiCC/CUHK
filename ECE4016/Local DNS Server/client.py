from socket import *
from dnslib import DNSRecord

serverIP = '127.0.0.1'
serverPort = 1234
domain = "baidu.com"

# 构建DNS查询
query = DNSRecord.question(domain)

# 发送查询
clientSocket = socket(AF_INET, SOCK_DGRAM)
clientSocket.sendto(query.pack(), (serverIP, serverPort))

# 接收并打印响应
response, _ = clientSocket.recvfrom(2048)
print(DNSRecord.parse(response))
