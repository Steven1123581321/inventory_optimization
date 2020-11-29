from collections import defaultdict

class Graph:
    
    def __init__(self, vertex):
        self.V = vertex
        self.graph = defaultdict(list)
        self.representitives = []
        self.members = []
        self.dictionary_sc = defaultdict(list)

    # Add edge into the graph
    def add_edge(self, s, d):
        self.graph[s].append(d)

    # dfs
    def dfs(self, d, visited_vertex):
        visited_vertex[d] = True
        for i in self.graph[d]:
            if not visited_vertex[i]:
                self.dfs(i, visited_vertex)

    def fill_order(self, d, visited_vertex, stack):
        visited_vertex[d] = True
        for i in self.graph[d]:
            if not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)
        stack = stack.append(d)

    # transpose the matrix
    def transpose(self):
        g = Graph(self.V)

        for i in self.graph:
            for j in self.graph[i]:
                g.add_edge(j, i)
        return g

    # Get stongly connected components in a list
    def get_scc(self):
        stack = []
        visited_vertex = [False] * (self.V)

        for i in range(self.V):
            if not visited_vertex[i]:
                self.fill_order(i, visited_vertex, stack)

        gr = self.transpose()

        visited_vertex = [False] * (self.V)

        while stack:
            i = stack.pop()
            if not visited_vertex[i]:
                self.representitives.append(i)
                gr.dfs(i, visited_vertex)
                members_list = [i for i, x in enumerate(visited_vertex) if x]
                self.members.append([item for item in members_list if item not in self.representitives])
                flat_list1 = [item for sublist in self.members for item in sublist]
                flat_list = [item for item in flat_list1 if item not in flat_list]
                if flat_list:
                    self.dictionary_sc[i] = flat_list
                flat_list = flat_list1