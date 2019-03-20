import numpy as np
import math


class Node:
    def __init__(self):
        self.value = None
        self.left_child = None
        self.right_child = None


class KDT:
    def __init__(self):
        self.root_node = None
        self.nearest_point = None
        self.min_distance = 999

    def create(self, T):
        root_node = self.cut(T, 0)
        self.root_node = root_node

    # 只有向下找初始点的功能
    def search(self, target):
        current_node = self.root_node
        level = 0
        while current_node.left_child is not None and current_node.right_child is not None:
            d = level % 2
            if target[d] < current_node.value[d]:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
            level += 1
        print('current nearest point: %s' % current_node.value)

    def find(self, target, current_node, level):
        if current_node.left_child is None and current_node.right_child is None:
            self.nearest_point = current_node.value
            self.min_distance = self.distance(target, self.nearest_point)
            print('initial nearest point: %s' % current_node.value)
            return

        d = level % 2
        if target[d] < current_node.value[d]:
            self.find(target, current_node.left_child, level+1)
        else:
            self.find(target, current_node.right_child, level+1)

        if target[d] < current_node.value[d]:
            # 说明向下遍历时去了左节点，回溯时找右节点
            if self.min_distance > self.distance(target, current_node.right_child.value):
                # 更新最近点
                self.min_distance = self.distance(target, current_node.right_child.value)
                self.nearest_point = current_node.right_child.value
        else:
            # 说明向下遍历时去了右节点，回溯时找左节点
            if self.min_distance > self.distance(target, current_node.left_child.value):
                # 更新最近点
                self.min_distance = self.distance(target, current_node.left_child.value)
                self.nearest_point = current_node.left_child.value

    def cut(self, T, level):
        if len(T) == 0:
            return
        # 不用加1了，因为维度从0开始计数，2维变量
        d = level % 2
        new_T = T[T[:, d].argsort()]
        # NOTICE 中位数只取高位
        index = int(len(new_T)/2)

        current_node = Node()
        current_node.value = new_T[index]
        # 生成新的一层
        current_node.left_child = self.cut(new_T[:index], level+1)
        current_node.right_child = self.cut(new_T[index+1:], level+1)
        return current_node

    def distance(self, X, Y):
        return math.sqrt(np.sum((X[i]-Y[i])**2 for i in range(len(X))))


T = np.array([[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]])
target = [3, 4.5]
tree = KDT()
tree.create(T)
tree.find(target, tree.root_node, 0)
print(tree.nearest_point)
print(tree.min_distance)
