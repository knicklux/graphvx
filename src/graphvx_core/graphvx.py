import sys
from copy import copy

from graphvx_core.hooks import *
from graphvx_core.vectorizer.cvxpy_vectorizer import *
from graphvx_core.vectorizer.numpy_vectorizer import *
from scipy.sparse import lil_matrix

X_NID: int = 0
X_OBJ: int = 1
X_VARS: int = 2
X_CON: int = 3
X_IND: int = 4
X_LEN: int = 5
X_DEG: int = 6
X_NEIGHBORS: int = 7
X_VECTIZER: int = 8

Z_EID: int = 0
Z_OBJ: int = 1
Z_CON: int = 2
Z_IVARS: int = 3
Z_ILEN: int = 4
Z_XIIND: int = 5
Z_ZIJIND: int = 6
Z_UIJIND: int = 7
Z_JVARS: int = 8
Z_JLEN: int = 9
Z_XJIND: int = 10
Z_ZJIIND: int = 11
Z_UJIIND: int = 12


class GraphVX:
    __default_objective = cp.norm(0)
    __default_constraints = []

    # Data Structures
    # ---------------
    # node_objectives  = {int NId : CVXPY Expression}
    # node_constraints = {int NId : [CVXPY Constraint]}
    # edge_objectives  = {(int NId1, int NId2) : CVXPY Expression}
    # edge_constraints = {(int NId1, int NId2) : [CVXPY Constraint]}
    # all_variables = set(CVXPY Variable)
    # graph = nx.DiGraph structure
    #
    # ADMM-Specific Structures
    # ------------------------
    # node_variables   = {int NId :
    #       [(CVXPY Variable id, CVXPY Variable name, CVXPY Variable, offset)]}
    # node_values = {int NId : numpy array}
    # node_values points to the numpy array containing the value of the entire
    #     variable space corresponding to then node. Use the offset to get the
    #     value for a specific variable.
    def __init__(self, graph, node_f, edge_f):

        # Initialize data structures
        self.node_to_id = {}
        self._nodes = {}
        self.node_objectives = {}
        self.node_variables = {}
        self.node_expressive_variables = {}
        self.node_constraints = {}
        self._edges = {}
        self.edge_objectives = {}
        self.edge_constraints = {}
        self.node_values = {}
        self.all_vars = []
        self.node_ids = []
        self.edge_ids = []
        # self.all_variables = list()
        self.status = None
        self.value = 0
        self.graph = None
        # self.edge_f = edge_f
        # self.node_f = node_f

        # values available after finalize:
        self.edge_list = None
        self.node_list = None
        self.A = None
        self.x_length = 0
        self.z_length = 0

        # Initialize graph
        self.graph = graph

        # evaluate edge_f and node_f to build the graph
        for node in self.graph.nodes:
            obj, constrs, vars, id = node_f(node)
            self.node_ids.append(id)
            self.node_to_id[node] = id
            self._nodes[id] = node

            self.node_objectives[id] = obj
            # variable = (CVXPY Variable id, CVXPY Variable name, CVXPY Variable, offset)
            # self.node_variables[id] = self.__ExtractVariableList(obj.variables())
            self.node_variables[id] = self.__ExtractVariableList(vars)
            self.node_expressive_variables[id] = vars
            self.node_constraints[id] = constrs
            self.all_vars += [var for var in vars]

        for edge in self.graph.edges:
            # recover id
            id_0 = self.node_to_id[edge[0]]
            id_1 = self.node_to_id[edge[1]]
            etup = (id_0, id_1)
            obj, constrs = edge_f(edge, self.node_expressive_variables[id_0], self.node_expressive_variables[id_1])
            # obj, constrs =
            # edge_f(edge, [v[2] for v in self.node_variables[id_0]], [v[2] for v in self.node_variables[id_1]])

            self.edge_objectives[etup] = obj
            self.edge_constraints[etup] = constrs
            self.edge_ids.append(etup)

    def build_etup(self, edge):
        left_id = self.lookup_node_id(edge[0])
        entered_id = self.lookup_node_id(edge[1])
        return left_id, entered_id

    def lookup_edge(self, etup):
        return self.edge_ids.index(etup)

    def lookup_node_id(self, node):
        return self.node_ids[list(self.graph.nodes).index(node)]

    def get_node_degree(self, node):
        # return self.graph.degree(node)
        return len(list(self.get_neighbors(node)))

    def get_neighbors(self, node):
        return self.graph.neighbors(node)

    def get_neighbor_id(self, node):
        return [self.node_to_id[n] for n in self.graph.neighbors(node)]

    # Simple iterator to iterator over all _nodes in graph. Similar in
    # functionality to Nodes() iterator of PUNGraph in Snap.py.
    def nodes(self):
        return self.graph.nodes

    # Simple iterator to iterator over all edge in graph. Similar in
    # functionality to Edges() iterator of PUNGraph in Snap.py.
    def edges(self):
        return self.graph.edges

    # Helper method to get CVXPY Variables out of a CVXPY variables
    def __ExtractVariableList(self, variables):
        l = [(var.name(), var) for var in variables]
        # Sort in ascending order by name
        l.sort(key=lambda t: t[0])
        l2 = []
        offset = 0
        for (varName, var) in l:
            # Add tuples of the form (id, name, object, offset)
            l2.append((var.id, varName, var, offset))
            temp = var.size[0] if hasattr(var.size, '__len__') else var.size
            offset += temp
        return l2

    # Helper method to get a tuple representing an edge. The smaller NId
    # goes first.
    def get_edge_tup(self, NId1, NId2):
        return (NId1, NId2) if NId1 < NId2 else (NId2, NId1)

    def finalize(self):

        # Organize information for each node in helper node_info structure
        node_info = {}
        # Keeps track of the current offset necessary into the shared node
        # values Array
        x_length = 0
        for nid in self.node_ids:
            ni = self._nodes[nid]
            deg = self.get_node_degree(ni)
            obj = self.node_objectives[nid]
            variables = self.node_variables[nid]
            con = copy(self.node_constraints[nid])
            neighbors = self.get_neighbor_id(ni)
            # Node's constraints include those imposed by _edges
            for neighbor in self.get_neighbors(ni):
                neighborid = self.lookup_node_id(neighbor)
                etup = (nid, neighborid)
                econ = copy(self.edge_constraints[etup])
                con += econ
            # Calculate sum of dimensions of all Variables for this node
            size = sum([var.size[0] if hasattr(var.size, '__len__') else var.size for (varID, varName, var, offset) in
                        variables])
            # Nearly complete information package for this node
            node_info[nid] = (nid, obj, variables, con, x_length, size, deg, list(neighbors))
            x_length += size
        self.x_length = x_length

            # Organize information for each node in final edge_list structure and
        # also helper edge_info structure
        edge_list = []
        edge_info = {}
        # Keeps track of the current offset necessary into the shared edge
        # values Arrays
        z_length = 0
        for etup in self.edge_ids:
            obj = copy(self.edge_objectives[etup])
            con = copy(self.edge_constraints[etup])

            con += self.node_constraints[etup[0]] + self.node_constraints[etup[1]]

            # Get information for each endpoint node
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            ind_zij = z_length
            ind_uij = z_length
            z_length += info_i[X_LEN]
            ind_zji = z_length
            ind_uji = z_length
            z_length += info_j[X_LEN]
            # Information package for this edge
            tup = (etup, obj, con, info_i[X_VARS], info_i[X_LEN], info_i[X_IND], ind_zij, ind_uij, info_j[X_VARS],
                   info_j[X_LEN], info_j[X_IND], ind_zji, ind_uji)
            edge_list.append(tup)
            edge_info[etup] = tup

        self.z_length = z_length
        self.edge_list = edge_list

        # Populate sparse matrix A.
        # A has dimensions (p, n), where p is the length of the stacked vector
        # of node variables, and n is the length of the stacked z vector of
        # edge variables.
        # Each row of A has one 1. There is a 1 at (i,j) if z_i = x_j.
        A = lil_matrix((z_length, x_length), dtype=np.int32)
        for etup in self.edge_ids:
            # print("new etup")
            # print(etup)
            info_edge = edge_info[etup]
            info_i = node_info[etup[0]]
            info_j = node_info[etup[1]]
            for offset in range(info_i[X_LEN]):
                # print(info_edge[Z_ZIJIND])
                row = info_edge[Z_ZIJIND] + offset
                col = info_i[X_IND] + offset
                # print("Bulidng A from i: {}, {}".format(row, col))
                A[row, col] = 1
            for offset in range(info_j[X_LEN]):
                # print(info_edge[Z_ZJIIND])
                row = info_edge[Z_ZJIIND] + offset
                col = info_j[X_IND] + offset
                # print("Bulidng A from j: {}, {}".format(row, col))
                A[row, col] = 1

        self.A = A

        # Create final node_list structure by adding on information for
        # node neighbors
        node_list = []
        for nid, info in node_info.items():
            entry = [nid, info[X_OBJ], info[X_VARS], info[X_CON], info[X_IND], info[X_LEN], info[X_DEG]]
            # Append information about z- and u-value indices for each
            # node neighbor
            for i in range(info[X_DEG]):
                neighborId = info[X_NEIGHBORS][i]
                indices = (Z_ZIJIND, Z_UIJIND) if nid < neighborId else (Z_ZJIIND, Z_UJIIND)
                # indices = (Z_ZIJIND, Z_UIJIND)
                einfo = edge_info[self.get_edge_tup(nid, neighborId)]
                entry.append(einfo[indices[0]])
                entry.append(einfo[indices[1]])
            node_list.append(entry)

        self.node_list = node_list

    # Insert into hash to support GetNodeValue()
    def save_solution(self, num_iters, max_iters, x):
        print("writing node_values[nid] for all nid")

        for entry in self.node_list:
            nid = entry[X_NID]
            index = entry[X_IND]
            # size = entry[X_LEN]
            vals = np.zeros((entry[X_LEN],))

            for (varID, varName, var, offset) in entry[X_VARS]:
                size = var.size[0] if hasattr(var.size, '__len__') else var.size
                vals[offset:offset + size] = x[index+offset:index+offset+size]
            self.node_values[nid] = vals

        if num_iters <= max_iters:
            self.status = 'Optimal'
        else:
            self.status = 'Incomplete: max iterations reached'
        self.value = self.get_total_problem_value()

    # API to get node Variable value after solving with ADMM.
    def get_node_value(self, id, Name):
        for (varID, varName, var, offset) in self.node_variables[id]:
            if varName == Name:
                offset = offset
                value = self.node_values[id]
                temp = var.size[0] if hasattr(var.size, '__len__') else var.size
                return value[offset:(offset + temp)]
        return None

    # Iterate through all variables and update values.
    # Sum all objective values over all _nodes and _edges.
    def get_total_problem_value(self):
        result = 0.0

        for nid in self.node_ids:
            for (varID, varName, var, offset) in self.node_variables[nid]:
                _, undo = cpVectorizer.auto(var)
                var.value = undo(self.get_node_value(nid, varName))

        for obj in self.node_objectives.values():
            result += obj.value

        for etup in self.edge_ids:
            result += self.edge_objectives[etup].value
        return result

    # Prints value of all node variables to console or file, if given
    def print_solution(self, Filename=None):
        out = sys.stdout if (Filename == None) else open(Filename, 'w+')

        out.write('Status: %s\n' % self.status)
        out.write('Total Objective: %f\n' % self.value)
        for ni in self.nodes():
            nid = ni.GetId()
            s = 'Node %d:\n' % nid
            out.write(s)
            for (varID, varName, var, offset) in self.node_variables[nid]:
                val = np.transpose(self.get_node_value(nid, varName))
                s = '  %s %s\n' % (varName, str(val))
                out.write(s)
