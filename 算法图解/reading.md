> [《算法图解》](https://book.douban.com/subject/26979890/)希望大家支持正版，下面是我在阅读完这本书之后做的一些总结。

## 第一章
### 算法简介
#### 二分查找
例如，想在电话簿中寻找以名字开头为K的人，我们如果从头开始找，那我们可能要翻阅半本电话簿才能找到，但我们知道K大概会在中间位置，那我们便可以在中间部分直接查找，在这种情况下，我们所使用的算法就是二分查找。

> 二分查找，也称折半查找、对数查找，是一种在有序数组中查找某一特定元素的查找演算法。
    
```
# 二分查找(while版本)
def binary_search(list, item):
    # item为目标元素
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
```
#### 运行时间
每次介绍算法时，我们都将讨论其运行时间，一般而言选择效率最高的算法，以最大限度地减少运行时间或者占用时间。
#### 大O表示法
`大O表示法`是一种特殊的表示法，指出了算法的速度有多块，通常情况下`大O表示法`指的是最糟情况下的运行时间。

下面列举一些常见的大O运行时间：
- `O(log  n)`，也叫对数时间，常见算法包括二分查找
- `O(n)`，也叫线性时间，常见算法包括简单查找
- `O(n *  log n)`，常见算法比如速度比较快的快速排序
- `O(n^2)`，常见算法包括选择排序
- `O(n!)`，常见算法包括旅行商问题

## 第二章
### 选择排序
#### 数组
数组内的所有元素都是需要紧密相连的，所以插入或者删除新的元素对原有数据的改动会比较大，但可以迅速的根据下标读取元素。
#### 链表
链表内的所有元素可能分布在不同的内存空间，他们之间通过指向进行连接，因此在插入或者删除元素的时候只需要改变指向就可以，但是想要读取此链上指定位置的元素要从头开始遍历。
#### 数组和链表的比较
/ |数组 | 链表
--- |---- | ---
读取|O(1) |  O(n)
插入|O(n) |  O(1)
删除|O(n) |  O(1)
#### 选择排序
> 选择排序是一种简单直观的排序算法。它的工作原理如下。首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。
```
# 选择排序
def find_smallest(list):
    smallest = list[0]
    smallest_index = 0
    for i in range(1, len(list)):
        if list[i] < smallest:
            smallest = list[i]
            smallest_index = i
    return smallest_index


def selection_sort(list):
    new_list = []
    for i in range(len(list)):
        smallest = find_smallest(list)
        new_list.append(list.pop(smallest))
    return new_list
```

## 第三章
### 递归
> 递归，又译为递回，在数学与计算机科学中，是指在函数的定义中使用函数自身的方法。
递归只是让解决方案更清晰，并没有性能上的优势。在有些情况下，使用循环的性能更好。
#### 基线条件和递归条件
由于递归函数调用自己，因此编写的函数容易导致无限循环。因此指定停止递归的条件就是基线条件，继续执行函数的条件就是递归条件。
#### 栈
栈为后进先出的一种数据结构，递归中会生成一系列的调用栈。
#### 阶乘递归

> 函数调用自身，称为递归。如果尾调用自身，就称为尾递归。
递归非常耗费内存，因为需要同时保存成千上百个调用记录，很容易发生"栈溢出"错误（stack overflow）。但对于尾递归来说，由于只存在一个调用记录，所以永远不会发生"栈溢出"错误。
```
# 递归
def factorial(x):
    if x == 1:
        return 1
    else:
        return x * factorial(x-1)

# 本书并没有写相应的尾递归内容，这里也是我自己加的
# 尾递归
def factorial_recursion(x, total):
    if x == 1:
        return total
    else:
        return factorial(x-1, x * total)
```

## 第四章
### 快速排序
快速排序使用分而治之的策略，对数据进行递归式排序。

#### 分而治之
分而治之是一种著名的递归式问题解决方法，具体步骤如下：
- 找出基线
- 不断将问题分解（或者说缩小规模），直到符合基线条件

#### 欧几里得算法
> 在数学中，辗转相除法，又称欧几里得算法，是求最大公约数的算法。
两个整数的最大公约数是能够同时整除它们的最大的正整数。辗转相除法基于如下原理：两个整数的最大公约数等于其中较小的数和两数的差的最大公约数。

#### 快速排序
##### 基线条件
基线条件为数组为空或只包含一个元素。
##### 基准值
快速排序需要对数组进行分解，因此需要一个基准值，以基准值对数组元素进行分区，一般选取数组中第一个元素为基准值。再在被分区的部分重复以上过程，最后可以得到排序结果。

```
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
```

#### 归并排序
简单来说就是先将数组不断细分成最小的单位，然后每个单位分别排序，排序完毕后合并，重复以上过程最后就可以得到排序结果。同样也是采用分而治之的思想。

```
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
```

当数据量越来越大时，

归并排序：比较次数少，速度慢。

快速排序：比较次数多，速度快。


## 第五章
### 散列表
#### 散列函数
需要满足的要求：
- 同一输入的输出必须一致
- 不同的输入映射到不同的索引
### 散列表应用
#### 查找
实现电话簿：
- 创建映射
- 查找

```
# 实现电话簿
def creat_phone_book():
    # 创建散列表
    phone_book = dict()
    phone_book["aa"] = 123456
    phone_book["bb"] = 654321
    # 查找
    print phone_book["aa"]
```

#### 防止重复
```
用来记录是否目标已经存在
# 投票防重复
voted = {}
def check_voted(name):
    if voted.get(name):
        print "已经存在"
    else:
        voted[name] = True
        print "请投票"
```

#### 缓存
实现快速响应，无需等待耗时处理
```
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
```

### 散列冲突
给两个键分配的位置相同，这种情况下就需要在这个位置上储存一个链表。

- 散列函数很重要，最理想的情况是散列函数均匀地映射到散列表的不同位置
- 如果散列表储存的链表过长，会导致散列表的速度急剧下降

/ |散列表（平均情况）|散列表（最糟情况）|数组 | 链表
--- |--- |---- |---- | ---
读取|O(1)|O(n)|O(1) |  O(n)
插入|O(1)|O(n)|O(n) |  O(1)
删除|O(1)|O(n)|O(n) |  O(1)

避免冲突：
- 使用低的装填因子
- 良好的散列函数

#### 装填因子
装填因子 = 散列表包含的元素数/位置总数
一旦装填因子变大，就需要在散列表中添加位置，这被称为调整长度。
一般装填因子大于0.7，就调整散列表的长度


## 第六章
### 广度优先搜索
#### 图
图模拟一组连接，一个图是表示物件与物件之间的关系的方法。
#### 广度优先搜索
广度优先搜索是一种用于图的查找算法。可解决如下问题：
- 从节点a出发，有前往节点b的路径嘛？
- 从节点a出发，前往节点b的哪条路径最短？
#### 队列
队列是一种先进先出的数据结构。

#### 实现广度优先搜索
```
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
```
#### 运行时间

广度优先搜索过程中的

队列时间是固定的即 `O(1 * 人数)`，

搜索过程中的时间为`O(边数)`，

因此广度优先搜索的运行时间为`O(人数 + 边数)`。

## 第七章
### 狄克斯特拉算法（dijkstra）
计算加权图最短路径且不适用于负权图
### 狄克斯特拉算法步骤
- 找出`权重`最小的节点，即可在最短时间内到达的节点
- 更新该节点的邻居的路径权重
- 重复这个步骤，直到对图中每一个节点都这么做
- 计算最终路径

#### 权重
狄克斯特拉算法用于每条边都有关联数字的图，这些数字称为`权重`。
#### 环
在无向图中每条边都是一个环，在有向图中从节点a开始走一圈又能回到节点a，这便是`环`。
#### 负权边
在图中有的边权为负数，这时就不能使用狄克斯特拉算法，这是因为dijkstra算法在计算最短路径时，不会因为负边的出现而更新已经计算过的顶点的路径长度，这样一来，在存在负边的图中，就可能有某些顶点最终计算出的路径长度不是最短的长度。

### 实现
![](https://user-gold-cdn.xitu.io/2018/2/26/161d0f089424a6f3)

```
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
```

## 第八章
### 贪婪算法
贪婪算法最大的优点就是简单易行，每步都采取最优的做法
#### 广播台覆盖问题
```
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
```
### NP完全问题
必须计算每个可能的集合

时间复杂度近似`O(2^n)`

- 涉及所有组合的问题通常是NP完全问题
- 元素较少时算法的运行速度非常快，但随着元素数量的增加，速度会变得非常慢
- 不能将问题分为小问题，必须考虑各种可能的情况，这可能是NP完全问题
- 如果问题涉及序列（如旅行商问题中城市序列）且难以解决，它可能就是NP完全问题
- 如果问题涉及集合（如广播台集合）且难以解决，它可能就是NP完全问题
- 如果问题可转化为集合覆盖问题或旅行商问题，那它肯定是NP完全问题


## 第九章
### 动态规划
#### 背包问题
```
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

print '最大值为：' + str(knapsack_dynamic(w, p, n, m, x))
print '物品的索引：', x
```

## 第十章
### K最近邻算法
#### 特征抽取
两个点之间的特征相似度可用毕达哥拉斯公式表示
#### 回归
K最近邻算法做两项基本工作——分类和回归：
- 分类就是编组
- 回归就是预测结果
#### 挑选合适的特征

## 总结
这本书总的来说就是一本算法入门书，在阅读完这本书之后，很多理念比之前更加容易理解，但是还有很多东西要理解，这里我也只是做了简单的记录，总之还算一次不错的阅读。

## 写在最后
这次所有的代码都在这里了[https://github.com/sosoneo/ReadingNotes](https://github.com/sosoneo/ReadingNotes)

