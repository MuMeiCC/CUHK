# ECE4016 Assigment 1
*Jiarui Chen  120090361*

## How to run this program
1. Use `cd` command to get into the directory
2. Use `python localDNSServer.py 0` or `python localDNSServer.py 1` command to run the program. The second parameter is 0 means using public DNS Server to search, and 1 means doing iterative search.
3. After the program starts, we can use *dig* command in terminal to send query to the local DNS Server. For example, `dig www.baidu.com @127.0.0.1 -p 1234`, or use `client.py` code to send request to the host. 

## How to stop this program
Type Ctrl+C in the keyboard. After receiving next query, the program will be stopped immediately.