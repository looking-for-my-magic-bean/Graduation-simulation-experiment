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
import KSP
import numpy as np

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

# 加入计算节点 节点数从15开始，每个交换节点上增加一个计算节点。加入对应的边，边数从44开始，每个计算节点对应往返两条边，由于计算中心靠近交换节点的原因，边的长度默认为100
linkmap[1][15] = (44, 100)
linkmap[15][1] = (45, 100)
linkmap[2][16] = (46, 100)
linkmap[16][2] = (47, 100)
linkmap[3][17] = (48, 100)
linkmap[17][3] = (49, 100)
linkmap[4][18] = (50, 100)
linkmap[18][4] = (51, 100)
linkmap[5][19] = (52, 100)
linkmap[19][5] = (53, 100)
linkmap[6][20] = (54, 100)
linkmap[20][6] = (55, 100)
linkmap[7][21] = (56, 100)
linkmap[21][7] = (57, 100)
linkmap[8][22] = (58, 100)
linkmap[22][8] = (59, 100)
linkmap[9][23] = (60, 100)
linkmap[23][9] = (61, 100)
linkmap[10][24] = (62, 100)
linkmap[24][10] = (63, 100)
linkmap[11][25] = (64, 100)
linkmap[25][11] = (65, 100)
linkmap[12][26] = (66, 100)
linkmap[26][12] = (67, 100)
linkmap[13][27] = (68, 100)
linkmap[27][13] = (69, 100)
linkmap[14][28] = (70, 100)
linkmap[28][14] = (71, 100)


def Generate_slice_service_array(n=14, level=5):
    random.seed(666)
    nums = [[0] * (14+n) for _ in range(n)]
    for i in range(n):
        for j in range(14 + i + 1, 14 + n):
            nums[i][j] = random.randint(1, level)
            nums[j - 14][i + 14] = nums[i][j]
    return nums


def get_node_importance(n, trafic_dis):
    temp_dict = {}
    for i in range(n):
        temp_dict[i+1] = sum(trafic_dis[i])
    return temp_dict


def get_node_degree(linkmap):
    temp_dict = {}
    for key, val in linkmap.items():
        temp_dict[key] = len(val)
    return temp_dict


nonuniform = False  # True #
# traffic distribution, when non-uniform traffic is considered
trafic_dis = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

slice_service = Generate_slice_service_array(14, 5)
trafic_dis += slice_service
# 获得含有业务重要度的字典
Importance_dict = get_node_importance(28, trafic_dis)
node_degree_dict = get_node_degree(linkmap)

prob = np.array(trafic_dis) / np.sum(trafic_dis)  # 既没有按行也没有按列，而是按照总数的归一化
LINK_NUM = 44 + 28
NODE_NUM = 14 + 14
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

# KSP算法算路，扩展为28个节点的， 10条
Candidate_Paths = KSP.calcualte_Candidate_Paths(linkmap, NODE_NUM, 10)


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


def restore(n, t_nodes_orders, c_nodes_orders, paths_order):
    if n < len(t_nodes_orders):
        normal_nodes.append(t_nodes_orders[n])
    if n < len(c_nodes_orders):
        normal_nodes.append(c_nodes_orders[n])
    for i in range(len(paths_order)-1, -1, -1):
        if paths_order[i][0] in normal_nodes and paths_order[i][1] in normal_nodes:
            temp = paths_order[i]
            paths_order.remove(temp)
            Fault_nodes.remove(temp)


def get_surround_nodes(node, normal_nodes, linkmap):
    num = 0
    for i in normal_nodes:
        if linkmap[node][i] is not None:
            num += 1
    return num


# ！！！！！！边的恢复思路不清晰
def get_surround_edge(node, normal_node, linkmap, fault_edge):
    candidate_edge = []
    for i in range(1, 15):
        if linkmap[node][i] is not None and ([node, i] in fault_edge or [i, node] in fault_edge):
            if i in normal_node:
                candidate_edge.append([node, i, -linkmap[node][i][1], 1] if node < i else [i, node, -linkmap[i][node][1], 1])
            else:
                candidate_edge.append([node, i, -linkmap[node][i][1], 0] if node < i else [i, node, -linkmap[i][node][1], 0])
    candidate_edge.sort(key=lambda x: (x[3], x[2], x[0]), reverse=True)
    candidate_edge_list = np.array(candidate_edge)[:, :2].tolist()
    while len(candidate_edge_list) < 2:
        candidate_edge_list.append(fault_edge.popleft())

    return candidate_edge_list[0], candidate_edge_list[1]   # 返回一个2*2的二维数组，每一行是候选的修复边


def greedy_algorithm(t_nodes_orders, c_nodes_orders, Fault_nodes, normal_nodes, Importance_dict, node_degree_dict, linkmap):
    temp_t_nodes_order = []
    temp_c_nodes_order = []

    fault_edge_order_t = []
    fault_edge_order_c = []
    fault_edge_order = []

    t_nodes_order_list = []
    t_nodes_order = t_nodes_orders.copy()
    c_nodes_order = c_nodes_orders.copy()
    fault_edge = Fault_nodes.copy()
    normal_node = normal_nodes.copy()

    # 云的恢复
    # 恢复云节点
    for i in c_nodes_order:
        temp_c_nodes_order.append([i, Importance_dict[i]])
    temp_c_nodes_order.sort(key=lambda x: (x[1], x[0]), reverse=True)
    c_nodes_order_list = [i[0] for i in temp_c_nodes_order]
    # 恢复云节点对应的边
    for j in c_nodes_order_list:
        for i in range(1, 15):
            if linkmap[i][j] is not None:
                restore_edge_c = [i, j]
                fault_edge.remove([i, j])
                fault_edge_order_c.append(restore_edge_c)

    # 网的恢复
    for i in range(len(t_nodes_order)):
        # 恢复传送节点
        for j in t_nodes_order:
            surround_nodes_num = get_surround_nodes(j, normal_node, linkmap)
            temp_t_nodes_order.append([j, node_degree_dict[j], surround_nodes_num])
        temp_t_nodes_order.sort(key=lambda x: (x[1], x[2], x[0]), reverse=True)
        restore_node = temp_t_nodes_order[0][0]  # 获得当前阶段的恢复节点
        temp_t_nodes_order = []                  # 清理暂时的传送节点序列
        normal_node.append(restore_node)         # 将恢复的传送节点加入到正常节点中
        t_nodes_order.remove(restore_node)       # 将恢复的传送节点从未回复的传送节点中移除
        t_nodes_order_list.append(restore_node)  # 将恢复的传送节点按照顺序加入到节点恢复顺序列表中去
        # 恢复传送边
        restore_edge_t1,  restore_edge_t2 = get_surround_edge(restore_node, normal_node, linkmap, fault_edge)
        fault_edge.remove(restore_edge_t1)
        fault_edge.remove(restore_edge_t2)
        fault_edge_order_t.append(restore_edge_t1)
        fault_edge_order_t.append(restore_edge_t2)

    # 总的恢复边的顺序
    for i in range(len(fault_edge_order_c)):
        fault_edge_order.append(fault_edge_order_c[i])
        fault_edge_order.append(fault_edge_order_t[(i*2)])
        fault_edge_order.append(fault_edge_order_t[(i*2)+1])

    while len(fault_edge_order) < len(Fault_nodes):
        fault_edge_order.append(fault_edge.popleft())

    return t_nodes_order_list, c_nodes_order_list, fault_edge_order


if __name__ == "__main__":

    random.seed(66)  # 设置随机数种子，保证每次仿真实验条件相同的情况下具有相同的输出，放在循环中保证每次循环都一样

    algorithm = "联合算法"  # 随机算法 贪婪算法 自创算法
    bp_arr = []
    bp_arr_all = []
    bp_arr_all_cumulative = []
    resource_util_all = []
    # 故障节点
    global Fault_nodes, normal_nodes, paths_order
    paths_order = deque([])
    t_nodes_orders = [4, 5, 6, 7, 8]
    c_nodes_orders = [18, 19, 20, 21, 22]
    Fault_nodes = deque([[1, 8], [2, 4], [3, 6], [4, 5], [4, 11], [5, 7], [5, 6], [6, 10], [6, 14], [7, 10], [7, 8], [8, 9],
                         [4, 18], [5, 19], [6, 20], [7, 21], [8, 22]])  # 双向队列
    normal_nodes = [1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 23, 24, 25, 26, 27, 28]
    print("原始通信节点-t_node：", t_nodes_orders)
    print("原始计算节点-c_node：", c_nodes_orders)
    print("原始路径-path：", Fault_nodes)

    # 故障节点恢复顺序
    if algorithm == "随机算法":

        random.shuffle(t_nodes_orders)
        random.shuffle(c_nodes_orders)
        random.shuffle(Fault_nodes)
        Fault_nodes_copy = Fault_nodes.copy()
        print("随机算法-t_node：", t_nodes_orders)
        print("随机算法-c_node：", c_nodes_orders)
        print("随机算法-path：", Fault_nodes)

    elif algorithm == "贪婪算法":
        t_nodes_orders, c_nodes_orders, Fault_nodes = greedy_algorithm(t_nodes_orders, c_nodes_orders, Fault_nodes, normal_nodes,
                                                                       Importance_dict, node_degree_dict, linkmap)
        Fault_nodes = deque(Fault_nodes)
        Fault_nodes_copy = Fault_nodes.copy()
        print("贪婪算法-t_node：", t_nodes_orders)
        print("贪婪算法-c_node：", c_nodes_orders)
        print("贪婪算法-path：", Fault_nodes)

    elif algorithm == "联合算法":
        t_nodes_orders = [6, 8, 7, 5, 4]
        c_nodes_orders = [20, 22, 21, 19, 18]
        Fault_nodes = deque([[6, 20], [6, 10], [6, 14], [8, 22], [1, 8], [8, 9], [7, 21], [7, 8], [7, 10], [5, 19], [5, 6], [5, 7], [4, 18], [4, 5], [2, 4],
                             [4, 11], [3, 6]])
        Fault_nodes_copy = Fault_nodes.copy()
        print("联合算法-t_node：", t_nodes_orders)
        print("联合算法-c_node：", c_nodes_orders)
        print("联合算法-path：", Fault_nodes)

    for ex in range(10):

        np.random.seed(666)  # 设置随机数种子，保证每次仿真实验条件相同的情况下具有相同的输出，放在循环中保证每次循环都一样

        # 从ex为1的时候才开始修复，第0轮保持不变
        if ex > 0:
            # 每一轮修复一个节点对 每一轮修复两个节点（一个计算节点一个边节点），三条节点边
            if len(Fault_nodes_copy) > 2:  # Fault_nodes_copy中还剩两个故障边恢复时
                paths_order.append(Fault_nodes_copy.popleft())
                paths_order.append(Fault_nodes_copy.popleft())
                paths_order.append(Fault_nodes_copy.popleft())
            elif len(Fault_nodes_copy) > 1:  # # Fault_nodes_copy中还剩一个故障边恢复时
                paths_order.append(Fault_nodes_copy.popleft())
                paths_order.append(Fault_nodes_copy.popleft())
            elif len(Fault_nodes_copy) > 0:
                paths_order.append(Fault_nodes_copy.popleft())
            restore(ex - 1, t_nodes_orders, c_nodes_orders, paths_order)

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
        normal_nodes.sort()
        print('normal_nodes:', normal_nodes)
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

