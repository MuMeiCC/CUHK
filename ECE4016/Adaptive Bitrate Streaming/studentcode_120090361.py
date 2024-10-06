import math

# Store the index of used bitrate of previous chunks
bitrate_used = []

def student_entrypoint(Measured_Bandwidth, Previous_Throughput, Buffer_Occupancy, Available_Bitrates, Video_Time, Chunk, Rebuffering_Time, Preferred_Bitrate ):
    bitrate = [int(key) for key in Available_Bitrates]
    bitrate.sort()
    size = [Available_Bitrates[key] for key in Available_Bitrates]
    size.sort()
    result_bitr = BOLA(bitrate,size,Buffer_Occupancy,Chunk,Measured_Bandwidth,Video_Time)
    return result_bitr


def utility(S,index):
    return math.log(S[index]/S[0])

def BOLA(bitrate,size,buffer_info,chunk_info,bandwidth,vt,weight=5):
    '''
    Input:
    bitrate: a list with all bitrates sorted from lowest to highest
    size: a list with all size values of corresponding bitrate
    buffer_info: a dict, which is Buffer_Occupancy in function student_entrypoint
    chunk_info: a dict, which is Chunk in function student_entrypoint
    bandwidth: an integer, bandwidth measured in previous segment
    vt: an integer, the current video time
    weight: the value of γp in BOLA Algorithm, default value is 5

    Output:
    bitrate[bitrate_used[num]]: the next bitrate value we will use
    '''
    global bitrate_used
    p = chunk_info['time']  # Length of the chunks
    num = int(chunk_info['current'])    # The No. of chunk handling now
    M = len(size)   # The number of bitrates

    Qmax = math.ceil(buffer_info['size']*8/(bitrate[0]*p))
    Q = math.ceil(buffer_info['time']/p)
    t = min(vt,(chunk_info['left']+num)*p-vt)
    dbs = math.ceil(min(Qmax,max(t/(2*p),3)))   # Dynamic buffer size (unit: segment)
    vd = (dbs-1)/(utility(size,M-1)+weight)

    i,val = 0,float('-inf')
    for m in range(M):
        j = (vd*utility(size,m)+vd*weight-Q)/size[m]
        if j>=val:
            val = j
            i = m
    bitrate_used.append(i)

    if bitrate_used[num]>bitrate_used[num-1] or num==0:
        opt = 0
        for m in range(M):
            if size[m]*8/p<=max(bandwidth,size[0]*8/p):
                opt = m
        if buffer_info['time']<=p:
            opt = 0
        elif opt>=bitrate_used[num]:
            opt = bitrate_used[num]
        elif num>0 and opt<bitrate_used[num-1]:
            opt = bitrate_used[num-1]
        else:
            opt += 1
        bitrate_used[num] = opt

    return bitrate[bitrate_used[num]]