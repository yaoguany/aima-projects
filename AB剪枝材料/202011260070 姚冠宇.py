import sys
max=sys.maxunicode
min=-sys.maxunicode
treelist=['A', ['B', ['D', ['H', ('O', 3), ('P', 20)], ['I', ('Q', 2), ('R', 10)]], ['E', ['J', ('S', 13)], ['K', ('T', 22), ('U', 1)]]], ['C', ['F', ['L', ('V', 2), ('W', 10)]], ['G', ['M', ('X', 2), ('Y', 5)], ['N', ('Z', 3)]]]]
emtpylist=[]#存储输出数组形式的树
class node():
    def __init__(self,L:list,depth=0,Return=None):
        self.name=L[0]#名字
        self.value=[min,max]#alpha beta值
        self.children=[]#孩子节点
        self.depth=depth#深度
        self.Return=Return#收益
        self.parent=None#父母节点
        for i in L:#生成树
            if i!=self.name:
                if type(i) != tuple:
                    self.children.append(node(i,self.depth+1))#递归生成
                else:
                    temp=node(i[0],self.depth+1)#递归到根节点
                    temp.value=[i[1],i[1]]
                    temp.Return=i[1]
                    self.children.append(temp)
    def pprint(self):#调试时用可以不看
        print(self.name,self.depth)
        if self.value!=[min,max]:
            print(self.value)
        for i in self.children:
            i.pprint()
            
    def Search(self,Name):#寻找名字叫Name的节点
        if self.name==Name:#找到了该节点
            self.lprint(emtpylist)#输出节点
            return#跳出递归
        else:
            for i in self.children:
                i.Search(Name)#递归查找
            
    
    def lprint(self,L:list):#list形式打印
        if self.depth!=4:#非第四层叶子节点
            temp=[]
            temp.append(self.name)
        else:
            temp=(self.name,self.Return)#叶子节点，需要保留回报值
        L.append(temp)
        for i in self.children:#递归生成列表
            i.lprint(temp)


    def DecisionMax(self):
        if self.depth!=4:#非叶子节点
            temp=self.children[0]
            for i in self.children:#未被剪枝过程访问（说明被剪枝），不进行搜索
                if i.Return==None:
                    i.Return=min
            for i in self.children:
                if i.Return>temp.Return:
                    temp=i
            return temp
        else:
            return
    
    def DecisionMin(self):
        if self.depth!=4:#非叶子节点
            temp=self.children[0]
            for i in self.children:
                if i.Return==None:
                    i.Return=max
            for i in self.children:
                if i.Return<temp.Return:
                    temp=i
            return temp
        else:      
            return
    def Decision(self):
        if self.depth%2==0:
            return self.DecisionMax()
        else:
            return self.DecisionMin()
        
    def Run(self):
        while self!=None:
            self.NodePrint()
            self= self.Decision()

    def ChangeValue(self,child):#对子节点和当前节点比较确定是否更改alpha beta的值
        if self.depth%2==0:#max节点
            if  child.Return>self.value[0]:
                self.value[0]=child.Return
                self.Return=child.Return
        else:#min节点
            if child.Return<self.value[1]:
                self.value[1]=child.Return
                self.Return=child.Return
            
    def NodePrint(self):#单独打印节点信息
        print(self.name,self.Return,self.value)
        
    def AlphaBetaSearch(self):#剪枝函数（递归调用）
        self.NodePrint()#表明访问了当前节点打印输出
        for i in self.children:#对子节点
            if i.value==[min,max]:#非叶子节点
                if self.value!=[min,max]:#继承父节点的alphabeta值
                    i.value=self.value
                    i.Return=self.Return
                i.AlphaBetaSearch()#进行剪枝
                self.ChangeValue(i)#剪枝完成后根节点对这个叶子节点求是否要改值
                self.NodePrint()#回到了根节点输出一下
                if self.value[0]>=self.value[1]:#如果改完值满足剪枝条件则剪枝
                    #i.NodePrint()
                    break
            
            else:#叶子节点
                i.NodePrint()#访问节点
                self.ChangeValue(i)#改根节点的值
                self.NodePrint()#回到了根节点
                if self.value[0]>=self.value[1]:#判断是否剪枝
                    break
                
                

print("原列表")
print(treelist)#原列表
t=node(treelist)#生成树
t.Search("B")#搜索节点打印
print("搜索节点打印")
print(emtpylist)#输出打印结果
emptylist=[]#初始化输出列表,方便之后搜索
print("剪枝过程：")
t.AlphaBetaSearch()#alpha-beta剪枝并输出剪枝过程
print("决策过程：")
t.Run()