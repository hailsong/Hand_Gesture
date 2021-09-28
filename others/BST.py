class Node:
    def __init__(self, val):
        self.val = val
        self.leftChild = None
        self.rightChild = None
class BST:
    def __init__(self):
        self.root = None

    def setRoot(self, val):
        self.root = Node(val)

    def find(self, val):
        if (self.findNode(self.root, val) is False):
            return False
        else:
            return True

    def findNode(self, currentNode : Node, val):
        # 노드 데이터 없음
        if (currentNode is None):
            return False
        # 타겟 값과 같은 val 가진 노드 발견
        elif (val == currentNode.val):
            return currentNode
        # 타겟보다 큰 노드 발견, 왼쪽 child Node로 재귀
        elif (val < currentNode.val):
            return self.findNode(currentNode.leftChild, val)
        # 타겟보다 작은 노드 발견, 오른쪽 child Node로 재귀
        else:
            return self.findNode(currentNode.rightChild, val)

    def insert(self, val):
        if (self.root is None):
            self.setRoot(val)
        else:
            self.insertNode(self.root, val)

    def insertNode(self, currentNode, val):
        if (val <= currentNode.val):
            if (currentNode.leftChild):
                self.insertNode(currentNode.leftChild, val)
            else:
                currentNode.leftChild = Node(val)
        elif (val > currentNode.val):
            if (currentNode.rightChild):
                self.insertNode(currentNode.rightChild, val)
            else:
                currentNode.rightChild = Node(val)

tree = BST()
