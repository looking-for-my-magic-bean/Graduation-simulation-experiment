from __future__ import division

import xlwt
from collections import defaultdict, deque
import struct
import types
import string
import numpy as np
import math
import copy
import random
import datetime

# in v1_4, we start to use continuous time simulation (still with a time granularity,
# but instead of the number of requests at each following the Poisson distribution,
# the inter-arrival time between requests follows the negative exponential distribution)
# therefore, we can remove the use flag_map from the input

# consider only 16 slots

# for load3, we set lambda_req = 8, for load 2, 5, for load 1, 3

# change topology, here below
from matplotlib import pyplot as plt

linkmap = defaultdict(lambda: defaultdict(lambda: None))  # Topology: NSFNet
linkmap[1][2] = (0, 1050)
linkmap[2][1] = (3, 1050)
linkmap[1][3] = (1, 1500)
linkmap[3][1] = (6, 1500)
linkmap[1][8] = (2, 2400)
linkmap[8][1] = (22, 2400)

linkmap[2][3] = (4, 600)
linkmap[3][2] = (7, 600)
linkmap[2][4] = (5, 750)
linkmap[4][2] = (9, 750)
linkmap[3][6] = (8, 1800)
linkmap[6][3] = (15, 1800)

linkmap[4][5] = (10, 600)
linkmap[5][4] = (12, 600)
linkmap[4][11] = (11, 1950)
linkmap[11][4] = (32, 1950)
linkmap[5][6] = (13, 1200)
linkmap[6][5] = (16, 1200)
linkmap[5][7] = (14, 600)
linkmap[7][5] = (19, 600)

linkmap[6][10] = (17, 1050)
linkmap[10][6] = (29, 1050)
linkmap[6][14] = (18, 1800)
linkmap[14][6] = (41, 1800)
linkmap[7][8] = (20, 750)
linkmap[8][7] = (23, 750)
linkmap[7][10] = (21, 1350)
linkmap[10][7] = (30, 1350)

linkmap[8][9] = (24, 750)
linkmap[9][8] = (25, 750)
linkmap[9][10] = (26, 750)
linkmap[10][9] = (31, 750)
linkmap[9][12] = (27, 300)
linkmap[12][9] = (35, 300)
linkmap[9][13] = (28, 300)
linkmap[13][9] = (38, 300)

linkmap[11][12] = (33, 600)
linkmap[12][11] = (36, 600)
linkmap[11][13] = (34, 750)
linkmap[13][11] = (39, 750)
linkmap[12][14] = (37, 300)
linkmap[14][12] = (42, 300)
linkmap[13][14] = (40, 150)
linkmap[14][13] = (43, 150)

nonuniform = False  # True #
# traffic distribution, when non-uniform traffic is considered
trafic_dis = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]]

prob = np.array(trafic_dis) / np.sum(trafic_dis)  # 既没有按行也没有按列，而是按照总数的归一化
LINK_NUM = 44
NODE_NUM = 14
SLOT_TOTAL = 100  # eon的频谱，要遵循频谱一致性和连续性，一个业务根据业务带宽大小决定占用几个连续的频谱，占满了的话，就算中间有些频谱缝隙，新的业务也加不进来了

N = 10  # number of paths each src-dest pair 源节点和目标节点对的路径数量
M = 1  # first M starting FS allocation positions are considered 这个变量代表什么？

kpath = 1  # = 1 SP-FF, = 5, KSP-FF 分别是最短路算法和K条最短路算法，K条最短路是K条最短路中随机选择一条最近的路
Bandwidth_start = 2  # 25
Bandwidth_end = 10   # 101


lambda_req = 12  # average number of requests per provisioning period, for uniform traffic, = 10, for nonuniform traffic = 16  12
# lambda_time = [5+2*x for x in range(6)] # average service time per request; randomly select one value from the list for each episode evaluated
lambda_time = [14]  # 25 for all jlt experiments
len_lambda_time = len(lambda_time)  # 这里服务时间都设为了1

# generate source and destination pairs
# for each src-dst pair, we calculate its cumulative probability based on the traffic distribution
Src_Dest_Pair = []
prob_arr = []
for ii in range(NODE_NUM):
    for jj in range(NODE_NUM):
        if ii != jj:
            prob_arr.append(prob[ii][jj])
            temp = []
            temp.append(ii + 1)
            temp.append(jj + 1)
            Src_Dest_Pair.append(temp)
num_src_dest_pair = len(Src_Dest_Pair)
prob_arr[-1] += 1 - sum(prob_arr)  # 这一步的作用是为了保证prob_arr的概率和为1


Candidate_Paths = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None)))  # Candidate_Paths[i][j][k]:the k-th path from i to j
fp = open('Src_Dst_Paths.dat', 'rb')
for ii in range(1, NODE_NUM * NODE_NUM + 1):  # NODE_NUM*NODE_NUM import precalculated paths (in terms of path_links)
    # temp_path = []
    if ii % NODE_NUM == 0:              # ii为14的整数倍时
        i = ii // NODE_NUM              # i为14的倍数
        j = (ii % NODE_NUM) + NODE_NUM  # j一直为14
    else:
        i = (ii // NODE_NUM) + 1        # i为14的倍数结果加q1
        j = ii % NODE_NUM               # j为1到14循环
    # i:[1,1,1,...,1,  2,2,2,...,2,  ......, 14,14,14,...,14]
    # j:[1,2,3,...,14, 1,2,3,...,14, ......, 1, 2, 3, ...,14]
    temp_num = []
    for tt in range(N):  # number of paths each src-dest pair 源节点和目标节点对的路径数量
        test = list(struct.unpack("i" * 1, fp.read(4 * 1)))
        # struct.unpack("i", bytes)解压一个二进制字符，这句话是将文件中读取的4个二进制字节解压为字符串，返回的是元组。注意，fp.read()是顺序读取，没循环一次接着上次的位置读取
        temp_num += test  # temp_num[0]: the node-num of path k

    if i != j:
        for k in range(N):
            temp_path = list(struct.unpack("i" * temp_num[k], fp.read(4 * temp_num[k])))
            Candidate_Paths[i][j][k] = temp_path
            # note, if there are less than N paths for certain src-dest pairs, then the last a few values of temp_num equate to '0'
fp.close()


# print(Candidate_Paths)


def _get_path(src, dst, Candidate_Paths, k):  # get path k of from src->dst
    if src == dst:
        print('error: _get_path()')
        path = []
    else:
        path = Candidate_Paths[src][dst][k]
        if path is None:
            return None
    return path


def calclink(p):  # map path to links
    path_link = []
    for a, b in zip(p[:-1], p[1:]):
        k = linkmap[a][b][0]
        path_link.append(k)
    return path_link


def get_new_slot_temp(slot_temp, path_link, slot_map):
    for i in path_link:
        for j in range(SLOT_TOTAL):
            slot_temp[j] = slot_map[i][j] & slot_temp[j]
    return slot_temp


# only used when we apply heuristic algorithms
def mark_vector(vector, default):
    le = len(vector)
    flag = 0
    slotscontinue = []
    slotflag = []

    ii = 0
    while ii <= le - 1:
        tempvector = vector[ii:le]
        default_counts = tempvector.count(default)
        if default_counts == 0:
            break
        else:
            a = tempvector.index(default)
            ii += a
            flag += 1
            slotflag.append(ii)
            m = vector[ii + 1:le]
            m_counts = m.count(1 - default)
            if m_counts != 0:
                n = m.index(1 - default)
                slotcontinue = n + 1
                slotscontinue.append(slotcontinue)
                ii += slotcontinue
            else:
                slotscontinue.append(le - ii)
                break
    return flag, slotflag, slotscontinue


def judge_availability(slot_temp, current_slots, FS_id):
    (flag, slotflag, slotscontinue) = mark_vector(slot_temp, 1)
    fs = -1
    fe = -1
    if flag > 0:
        n = len(slotscontinue)
        flag_availability = 0  # Initialized to be unavailable
        t = 0
        for i in range(n):
            if slotscontinue[i] >= current_slots:
                if t == FS_id:
                    fs = slotflag[i]
                    fe = slotflag[i] + current_slots - 1
                    flag_availability = 1
                    return flag_availability, fs, fe
                t += 1
        return flag_availability, fs, fe
    else:
        flag_availability = 0
    return flag_availability, fs, fe


# update slotmap, mark allocated FS' as occupied
def update_slot_map_for_committing_wp(slot_map, current_wp_link, current_fs, current_fe, slot_map_t, current_TTL):
    for ll in current_wp_link:
        for s in range(current_fs, current_fe + 1):
            if slot_map[ll][s] != 1 or slot_map_t[ll][s] != 0:  # means error
                print('Error--update_slot_map_for_committing_wp!')
            else:  # still unused
                slot_map[ll][s] = 0
                slot_map_t[ll][s] = current_TTL
    return slot_map, slot_map_t


def update_slot_map_for_releasing_wp(slot_map, current_wp_link, current_fs, current_fe):  # update slotmap, mark released FS' as free
    for ll in current_wp_link:
        for s in range(current_fs, current_fe + 1):
            if slot_map[ll][s] != 0:  # this FS should be occupied by current request, !=0 means available now, which is wrong
                print('Error--update_slot_map_for_releasing_wp!')
            else:  # still unused
                slot_map[ll][s] = 1
    return slot_map


def release(slot_map, request_set, slot_map_t, time_to):  # update slotmap to release FS' occupied by expired requests
    if request_set:
        # update slot_map_t
        for ii in range(LINK_NUM):
            for jj in range(SLOT_TOTAL):
                if slot_map_t[ii][jj] > time_to:
                    slot_map_t[ii][jj] -= time_to
                elif slot_map_t[ii][jj] > 0:
                    slot_map_t[ii][jj] = 0
        #
        del_id = []
        for rr in request_set:
            request_set[rr][3] -= time_to  # request_set[rr][3] is TTL
            if request_set[rr][3] <= 0:
                current_wp_link = request_set[rr][0]
                fs_wp = request_set[rr][1]
                fe_wp = request_set[rr][2]
                # release slots on the working path of the request
                slot_map = update_slot_map_for_releasing_wp(slot_map, current_wp_link, fs_wp, fe_wp)
                del_id.append(rr)
        for ii in del_id:
            del request_set[ii]
    return slot_map, request_set, slot_map_t


def cal_len(path):
    path_len = 0
    for a, b in zip(path[:-1], path[1:]):
        path_len += linkmap[a][b][1]
    return path_len


def cal_FS(bandwidth, path_len):
    if path_len <= 625:
        num_FS = math.ceil(current_bandwidth / (4 * 12.5)) + 1  # 1 as guard band FS 向上取整
    elif path_len <= 1250:
        num_FS = math.ceil(current_bandwidth / (3 * 12.5)) + 1
    elif path_len <= 2500:
        num_FS = math.ceil(current_bandwidth / (2 * 12.5)) + 1
    else:
        num_FS = math.ceil(current_bandwidth / (1 * 12.5)) + 1
    return int(num_FS)


#  将数据写入新文件
def data_write(file_path, datas):
    f = xlwt.Workbook()
    sheet1 = f.add_sheet(u'sheet1', cell_overwrite_ok=True)  # 创建sheet

    # 将数据写入第 i 行，第 j 列
    i = 0
    for j in range(len(datas)):
        sheet1.write(i, j, datas[j])

    f.save(file_path)  # 保存文件


def restore(n, nodes_order, paths_order):
    if n < len(nodes_order):
        normal_nodes.append(nodes_order[n])
    for i in range(len(paths_order)-1, -1, -1):
        if paths_order[i][0] in normal_nodes and paths_order[i][1] in normal_nodes:
            temp = paths_order[i]
            paths_order.remove(temp)
            Fault_nodes.remove(temp)
def tanlan(nodes_order, paths_order):
    pass



if __name__ == "__main__":

    random.seed(66)  # 设置随机数种子，保证每次仿真实验条件相同的情况下具有相同的输出，放在循环中保证每次循环都一样

    algorithm = "贪婪算法"  # 随机算法 贪婪算法 自创算法
    bp_arr = []
    bp_arr_all = []
    bp_arr_all_cumulative = []
    resource_util_all = []
    # 故障节点
    global Fault_nodes, normal_nodes, paths_order
    paths_order = deque([])
    nodes_order = [4, 5, 6, 7, 8]
    Fault_nodes = deque([[1, 8], [2, 4], [3, 6], [4, 5], [4, 11], [5, 7], [5, 6], [6, 10], [6, 14], [7, 10], [7, 8], [8, 9]])  # 双向队列
    normal_nodes = [1, 2, 3, 9, 10, 11, 12, 13, 14]
    print("原始节点-node：", nodes_order)
    print("原始路径-path：", Fault_nodes)
    # 故障节点恢复顺序
    if algorithm == "随机算法":
        random.shuffle(nodes_order)
        random.shuffle(Fault_nodes)
        Fault_nodes_copy = Fault_nodes.copy()
        print("随机算法-node：", nodes_order)
        print("随机算法-path：", Fault_nodes)

    elif algorithm == "贪婪算法":

        Fault_nodes_copy = Fault_nodes.copy()
        # nodes_order = [6, 7, 4, 5, 8]
        # Fault_nodes = deque([[3, 6], [6, 10], [5, 6], [6, 14], [7, 8], [7, 10], [5, 7], [2, 4], [4, 5], [4, 11], [1, 8], [8, 9]])
        nodes_order = [6, 4, 5, 7, 8]
        Fault_nodes = deque([[6, 10], [6, 14], [2, 4], [4, 11], [4, 5], [5, 6], [5, 7], [7, 10], [7, 8], [8, 9], [1, 8], [3, 6]])
        print("贪婪算法-node：", nodes_order)
        print("贪婪算法-path：", Fault_nodes)

    for ex in range(10):

        np.random.seed(666)  # 设置随机数种子，保证每次仿真实验条件相同的情况下具有相同的输出，放在循环中保证每次循环都一样

        # 每一轮修复一个节点对
        if len(Fault_nodes_copy) > 1 and ex > 0:  # Fault_nodes_copy中还剩两个故障边恢复时
            paths_order.append(Fault_nodes_copy.popleft())
            paths_order.append(Fault_nodes_copy.popleft())
            restore(ex-1, nodes_order, paths_order)
        elif len(Fault_nodes_copy) > 0 and ex > 0:  # # Fault_nodes_copy中还剩一个故障边恢复时
            paths_order.append(Fault_nodes_copy.popleft())
            restore(ex - 1, nodes_order, paths_order)


        # initiate the EON
        slot_map = [[1 for x in range(SLOT_TOTAL)] for y in range(LINK_NUM)]  # Initialized to be all available
        slot_map_t = [[0 for x in range(SLOT_TOTAL)] for y in range(LINK_NUM)]  # the time each FS will be occupied

        service_time = lambda_time[np.random.randint(0, len_lambda_time)]  # 前面设置len_lambda_time为1，
        lambda_intervals = 1 / lambda_req  # average time interval between request

        request_set = {}

        req_id = 0
        num_blocks = 0

        time_to = 0
        num_req_measure = 10000
        resource_util = []

        while req_id < num_req_measure + 3000:

            (slot_map, request_set, slot_map_t) = release(slot_map, request_set, slot_map_t, time_to)

            time_to = 0
            while time_to == 0:
                time_to = np.random.exponential(lambda_intervals)

            if True:  # If is used just for the sake of convenience...

                req_id += 1

                # generate current request
                if nonuniform is True:
                    sd_onehot = [x for x in range(num_src_dest_pair)]
                    sd_id = np.random.choice(sd_onehot, p=prob_arr)
                    temp = Src_Dest_Pair[sd_id]
                else:
                    temp = Src_Dest_Pair[np.random.randint(0, num_src_dest_pair)]
                current_src = temp[0]
                current_dst = temp[1]
                current_bandwidth = np.random.randint(Bandwidth_start, Bandwidth_end)
                current_TTL = 0
                while current_TTL == 0 or current_TTL >= service_time * 2:
                    current_TTL = np.random.exponential(service_time)

                #  start provision the request
                blocking = 0

                for rr in range(kpath):

                    path_id = rr // M  # path to use
                    FS_id = math.fmod(rr, M)  # 取余数 the FS_id's available FS-block to use

                    path = _get_path(current_src, current_dst, Candidate_Paths, path_id)  # 最短路径[2,4,5,7]
                    # 判断路径中是否有故障节点，有的话直接阻塞+1
                    for i in range(len(path)-1):
                        temp = path[i:i+2]
                        if temp in Fault_nodes or temp.reverse() in Fault_nodes:
                            blocking = 1
                            break
                    if blocking == 1:
                        break

                    path_len = cal_len(path)  # physical length of the path
                    num_FS = cal_FS(current_bandwidth, path_len)
                    slot_temp = [1] * SLOT_TOTAL
                    path_links = calclink(path)
                    slot_temp = get_new_slot_temp(slot_temp, path_links,
                                                  slot_map)  # spectrum utilization on the whole path 确保新来的业务有可用的频谱资源
                    (flag, fs_start, fs_end) = judge_availability(slot_temp, num_FS, FS_id)
                    if flag == 1:
                        slot_map, slot_map_t = update_slot_map_for_committing_wp(slot_map, path_links, fs_start, fs_end,
                                                                                 slot_map_t,
                                                                                 current_TTL)  # update slotmap
                        temp_ = []  # update in-service requests
                        temp_.append(list(path_links))
                        temp_.append(fs_start)
                        temp_.append(fs_end)
                        temp_.append(current_TTL)
                        request_set[req_id] = temp_
                        break
                    elif rr == kpath - 1:
                        blocking = 1

                if req_id > 3000:
                    num_blocks += blocking  # count the number of requests that are blocked
                    resource_util.append((1 - np.sum(slot_map) / (LINK_NUM * SLOT_TOTAL)) * 100)
                    # print('Blocking Nums = {:d}'.format(num_blocks))

        bp = num_blocks / num_req_measure * 100
        bp_arr.append(bp)
        print('epoch:', ex + 1)
        print('Fault_nodes:', Fault_nodes)
        print('Blocking Probability = {:.2f}%'.format(bp))
        print('Blocking Probability Cumulative = {:.2f}%'.format(np.sum(bp_arr)))
        print('Mean Resource Utilization = {:.2f}%'.format(np.mean(resource_util)))
        bp_arr_all.append(bp)
        bp_arr_all_cumulative.append(np.sum(bp_arr))
        resource_util_all.append(np.mean(resource_util))

    # 存储
    np.save('data' + '/' + algorithm + '_bp_arr_all_' + str(lambda_req), bp_arr_all)
    np.save('data' + '/' + algorithm + '_bp_arr_all_cumulative_' + str(lambda_req), bp_arr_all_cumulative)
    np.save('data' + '/' + algorithm + '_resource_util_all_' + str(lambda_req), resource_util_all)
    data_write('data' + '/' + algorithm + '_bp_arr_all_' + str(lambda_req) + '.xlsx', bp_arr_all)
    data_write('data' + '/' + algorithm + '_bp_arr_all_cumulative_' + str(lambda_req) + '.xlsx', bp_arr_all_cumulative)
    data_write('data' + '/' + algorithm + '_resource_util_all_' + str(lambda_req) + '.xlsx', resource_util_all)

