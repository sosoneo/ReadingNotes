# -*- coding: UTF-8 -*-
from collections import deque


# chapter 1
# 二分查找(while版本)
def binary_search(list, item):
    # 用于根据要在其中查找的列表部分
    low = 0
    high = len(list) - 1
    while low <= high:
        # 只要范围没有缩小到只包含一个元素，就继续执行
        mid = (low + high) / 2
        # 检查中间元素
        guess = list[mid]
        if guess == item:
            # 找到了目标元素
            return mid
        if guess > item:
            # 猜的元素大了
            high = mid - 1
        else:
            # 猜的元素小了
            low = mid + 1
    # 没有找到目标元素
    return None


# 本书并没有将递归的相关内容，这是我在看到后面的递归内容回头补上的
# 二分查找(递归版本)
def binary_search_recursion(list, low, high, item):
    if low > high:
        # 没有找到目标元素
        return -1
    mid = low + (high - low) / 2
    if list[mid] > item:
        # 猜的元素大了
        return binary_search_recursion(list, low, mid - 1, item)
    if list[mid] < item:
        # 猜的元素笑了
        return binary_search_recursion(list, mid + 1, high, item)
    return mid


# chapter 2
# 选择排序
def find_smallest(list):
    # 储存最小的值
    smallest = list[0]
    # 储存最小元素的索引
    smallest_index = 0
    for i in range(1, len(list)):
        if list[i] < smallest:
            smallest = list[i]
            smallest_index = i
    return smallest_index


def selection_sort(list):
    new_list = []
    for i in range(len(list)):
        # 找出当前数组中最小的元素，并加入到新数组中
        smallest = find_smallest(list)
        new_list.append(list.pop(smallest))
    return new_list


# chapter 3
# 递归
def factorial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)


# 尾递归
def factorial_recursion(x, total):
    if x == 1:
        return total
    else:
        return factorial(x-1, x * total)


# chapter 4
# 快速排序
def quick_sort(array):
    if len(array) < 2:
        # 基线条件
        return array
    else:
        # 递归条件
        # 基准值
        pivot = array[0]
        # 分区
        less = [i for i in array[1:] if i <= pivot]
        greater = [i for i in array[1:] if i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)


# 本书只是稍稍提到了相关内容，并为对此进行详细展开，这里是笔者自己加上的内容
# 归并排序
def merge_list(list_a, list_b):
    # 合并数组
    new_list = []
    index_a = 0
    index_b = 0
    while index_a < len(list_a) and index_b < len(list_b):
        if list_a[index_a] < list_b[index_b]:
            new_list.append(list_a[index_a])
            index_a += 1
        else:
            new_list.append(list_b[index_b])
            index_b += 1
    while index_a < len(list_a):
        new_list.append(list_a[index_a])
        index_a += 1
    while index_b < len(list_b):
        new_list.append(list_b[index_b])
        index_b += 1
    return new_list


def merge_sort(array, low, high):
    # low 初始下标
    # high 结尾下标
    new = []
    if low < high:
        mid = (low + high) / 2
        # 递归排序最小单位数组
        merge_sort(array, low, mid)
        merge_sort(array, mid + 1, high)
        # 归并数组
        list_a = array[low:mid+1]
        list_b = array[mid+1:high+1]
        new = merge_list(list_a, list_b)
        start = low
        for i in new:
            array[start] = i
            start += 1
    return array


# chapter 5
# 实现电话簿
def creat_phone_book():
    # 创建散列表
    phone_book = dict()
    phone_book["aa"] = 123456
    phone_book["bb"] = 654321
    # 查找
    print phone_book["aa"]

# 投票防重复
voted = {}


def check_voted(name):
    if voted.get(name):
        print "已经存在"
    else:
        voted[name] = True
        print "请投票"

# 实现缓存
cache = {}


def get_data(url):
    if cache.get(url):
        return cache[url]
    else:
        data = "这里进行获取数据的耗时操作"
        # 完成之后进行缓存
        cache[url] = data
        return data


# chapter 6
# 创建图
graph = dict()
graph["you"] = ["alice", "bob", "claire"]
graph["bob"] = ["anuj", "peggy"]
graph["alice"] = ["peggy"]
graph["claire"] = ["tom", "jonny"]
graph["anuj"] = []
graph["peggy"] = []
graph["tom"] = []
graph["jonny"] = []


# 判断这个人是不是商人
def person_is_seller(name):
    return name[-1] == "m"


# 广度优先搜索
def bfs(name):
    # 创建搜索队列
    search_queue = deque()
    search_queue += graph[name]
    # 已经搜索过的节点数组， 防止无限循环
    searched = []
    while search_queue:
        person = search_queue.popleft()
        if person not in searched:
            if person_is_seller(person):
                print "找到商人 %s" % person
                return True
            else:
                search_queue += graph[person]
                searched.append(person)
    return False


# chapter 7
# 创建图
graph = {}
graph["start"] = {}
graph["start"]["a"] = 6
graph["start"]["b"] = 2
graph["a"] = {}
graph["a"]["fin"] = 1
graph["b"] = {}
graph["b"]["a"] = 3
graph["b"]["fin"] = 5
graph["fin"] = {}

# 创建开销
infinity = float("inf")
costs = {}
costs["a"] = 6
costs["b"] = 2
costs["fin"] = infinity

# 创建父节点
parents = {}
parents["a"] = "start"
parents["b"] = "start"
parents["fin"] = None

# 已经确定的节点，防止无限循环
processed = []


# 寻找权重最小的节点
def find_lowest_cost_node(costs):
    # 最小的花费
    lowest_cost = float("inf")
    # 最小花费的节点
    lowest_cost_node = None
    for node in costs:
        cost = costs[node]
        if cost < lowest_cost and node not in processed:
            lowest_cost = cost
            lowest_cost_node = node
    return lowest_cost_node


# dijkstra算法
def dijkstra():
    node = find_lowest_cost_node(costs)
    while node is not None:
        cost = costs[node]
        neighbors = graph[node]
        for n in neighbors.keys():
            new_cost = cost + neighbors[n]
            if costs[n] > new_cost:
                costs[n] = new_cost
                parents[n] = node
        processed.append(node)
        node = find_lowest_cost_node(costs)
    print costs
    print parents
    print processed


# chapter 8
# 广播台覆盖问题
def greedy():
    states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])
    stations = {}
    stations["kone"] = set(["id", "nv", "ut"])
    stations["ktwo"] = set(["wa", "id", "mt"])
    stations["kthree"] = set(["or", "nv", "ca"])
    stations["kfour"] = set(["nv", "ut"])
    stations["kfive"] = set(["ca", "az"])
    final_stations = set()
    while states_needed:
        best_station = None
        states_covered = set()
        for station, states in stations.items():
            covered = states_needed & states
            if len(covered) > len(states_covered):
                best_station = station
                states_covered = covered
        states_needed -= states_covered
        final_stations.add(best_station)
    print final_stations

# chapter 9
# 背包问题
# 这里使用了图解中的吉他，音箱，电脑，手机做的测试，数据保持一致
w = [0, 1, 4, 3, 1]   #n个物体的重量(w[0]无用)
p = [0, 1500, 3000, 2000, 2000]   #n个物体的价值(p[0]无用)
n = len(w) - 1   #计算n的个数
m = 4   #背包的载重量

x = []   #装入背包的物体，元素为True时，对应物体被装入(x[0]无用)
v = 0
#optp[i][j]表示在前i个物体中，能够装入载重量为j的背包中的物体的最大价值
optp = [[0 for col in range(m + 1)] for raw in range(n + 1)]
#optp 相当于做了一个n*m的全零矩阵的赶脚，n行为物件，m列为自背包载重量

def knapsack_dynamic(w, p, n, m, x):
    #计算optp[i][j]
    for i in range(1, n + 1):       # 物品一件件来
        for j in range(1, m + 1):   # j为子背包的载重量，寻找能够承载物品的子背包
            if (j >= w[i]):         # 当物品的重量小于背包能够承受的载重量的时候，才考虑能不能放进去
                optp[i][j] = max(optp[i - 1][j], optp[i - 1][j - w[i]] + p[i])    # optp[i - 1][j]是上一个单元的值， optp[i - 1][j - w[i]]为剩余空间的价值
            else:
                optp[i][j] = optp[i - 1][j]

    #递推装入背包的物体,寻找跳变的地方，从最后结果开始逆推
    j = m
    for i in range(n, 0, -1):
        if optp[i][j] > optp[i - 1][j]:
            x.append(i)
            j = j - w[i]

    #返回最大价值，即表格中最后一行最后一列的值
    v = optp[n][m]
    return v

if __name__ == '__main__':
    print binary_search_recursion([1, 3, 4, 5, 9], 0, 5, 3)

    print '最大值为：' + str(knapsack_dynamic(w, p, n, m, x))
    print '物品的索引：', x
