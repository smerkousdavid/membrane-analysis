from typing import List
from analysis.statistics import Statistics, compute_statistics
import numpy as np
import h5py as hf
import copy


class AnalysisData(object):
    def __init__(self, file_name: str, dtype=np.double):
        self.data = np.array([], dtype=dtype)
        self.name = file_name
        self.stats = None

    def get_data(self) -> np.ndarray:
        return self.data

    def get_name(self) -> str:
        return self.name

    def set_stats(self, stats: Statistics):
        self.stats = stats

    def get_stats(self) -> Statistics:
        if self.data and self.stats is None:
            self.stats = compute_statistics(self.data)
        return self.stats


# def HFAnalysisData(object):
#     def __init__(self, group: hf.Group):



class AnalysisNode(object):
    def __init__(self):
        self.nodes = []
        self.stats = None

    def add_node(self, node: AnalysisNode):
        self.nodes.append(node)

    def get_nodes(self) -> List[AnalysisNode]:
        return self.nodes
    
    def get_recursive_nodes(self, _ref=None) -> List[AnalysisNode]:
        if _ref is None:
            _ref = self.nodes
            if _ref is None:
                _ref = []
        copy_nodes = copy.copy(_ref)
        for node in self.nodes:
            copy_nodes.extend(node.get_recursive_nodes(_ref=node.nodes))
        return copy_nodes

    def get_all_analysis_objects(self) -> List[AnalysisData]:
        of_type = lambda item: isinstance(item, AnalysisData)
        all_analysis = list(filter(of_type, self.get_recursive_nodes(self.nodes)))
        return all_analysis
    
    def get_all_analysis_data(self) -> np.ndarray:
        all_analysis = self.get_all_analysis_objects()
        all_data = np.concatenate([analysis.get_data() for analysis in all_analysis])
        return all_data

    def calc_stats(self):
        all_data = self.get_all_analysis_data()
        self.stats = compute_statistics(all_data)

    def calc_recursive_stats(self, min_depth: int=0, max_depth: int=-1, _depth: int=0):
        if max_depth != -1 and _depth == max_depth:
            return  # nothing else to do
        
        of_type = lambda item: isinstance(item, AnalysisNode)
        node_types = list(filter(of_type, self.nodes))

        # iter through each node
        for node in node_types:
            node.calc_recursive_stats(min_depth, max_depth, _depth + 1)

            # make sure we're at our min depth
            if _depth >= min_depth:
                node.calc_stats()

    def get_stats(self):
        return self.stats
