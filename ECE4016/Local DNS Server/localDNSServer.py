# Jiarui Chen 120090361

from sys import argv
from socket import *
from dnslib import *

# IPs
serverIP = '127.0.0.1'
serverPort = 1234
publicIP = '10.20.232.47'
rootIP = '192.33.4.12'

# Query function
def query(reqIP, reqPort, req):
	print("Querying server: %s" %(reqIP))
	querySocket = socket.socket(AF_INET, SOCK_DGRAM)
	querySocket.connect((reqIP,reqPort))
	querySocket.send(req)
	result = querySocket.recv(2048)
	querySocket.close()
	return result

# Deal with query using public DNS Server
def publicReq(req):
	result = query(publicIP,53,req)
	return result

# Deal with query using iterative search
def iterQuery(req):
	IP = rootIP
	recs = []
	while True:
		mesg = query(IP,53,req)
		dnsRec = DNSRecord.parse(mesg)
		if len(dnsRec.rr)==0:
			if len(dnsRec.ar) > 0:
				IP = str(dnsRec.ar[0].rdata)
			else:
				if len(dnsRec.auth) > 0 and dnsRec.auth[0].rtype == 2:
					nsAddr = str(dnsRec.auth[0].rdata)
					req = DNSRecord.question(nsAddr).pack()
					recs.append(DNSRecord.parse(mesg))
					continue
		else:
			if dnsRec.rr[0].rtype==5:	#CNAME Record
				cnameAddr = str(dnsRec.rr[0].rdata)
				req = DNSRecord.question(cnameAddr).pack()
				recs.append(DNSRecord.parse(mesg))
				IP = rootIP
			elif dnsRec.rr[0].rtype==1:	#A Record
				recs.append(DNSRecord.parse(mesg))
				rec = recs[0]
				for i in range(1,len(recs)):
					rec.add_answer(*recs[i].rr)
				return rec.pack()

def localDNSServer(mode):
	# Create a socket for DNS Server
	serverSocket = socket.socket(AF_INET, SOCK_DGRAM)
	serverSocket.bind((serverIP,serverPort))
	serverCache = {}	# Using dict to cache
	while True:
		try:
			req,addr = serverSocket.recvfrom(2048)
			ques = DNSRecord.parse(req).questions[0].qname
			if ques in serverCache:	# If the request ip is in cache
				opt = serverCache[ques]
				parsed = DNSRecord.parse(opt)
				parsed.header.id = DNSRecord.parse(req).header.id
				reply = parsed.pack()
			else:
				if mode=="0":
					reply = publicReq(req)
					serverCache[ques] = reply
				else:
					reply = iterQuery(req)
					serverCache[ques] = reply
			serverSocket.sendto(reply,addr)
		except:
			print("The server stopped")
			break

def main():
	if len(argv)<2:
		print("Wrong input!")
	else:
		if argv[1]=="0" or argv[1]=="1":
			localDNSServer(argv[1])
		else:
			print("Wrong input!")


if __name__ == '__main__':
	main()
