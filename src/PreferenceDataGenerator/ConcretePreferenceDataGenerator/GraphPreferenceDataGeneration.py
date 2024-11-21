from itertools import combinations, product
from typing import override

import networkx as nx
import numpy as np
import torch

from src.DataStructures.ConcreteDataStructures.ActionData import ActionData
from src.DataStructures.ConcreteDataStructures.ActionPairsData import ActionPairsData
from src.DataStructures.ConcreteDataStructures.PreferencePairsData import (
    PreferencePairsData,
)
from src.FeedbackSource.AbsFeedbackSource import AbsFeedbackSource
from src.PreferenceDataGenerator.AbsPreferenceDataGenerator import (
    AbsPreferenceDataGenerator,
)


class GraphPreferenceDataGeneration(AbsPreferenceDataGenerator):
    """Base class incupsulating the required logic for generating preference data
    based on constructed graphs of preference relationships between
    actions
    """

    def __init__(
        self, feedbackSource: AbsFeedbackSource, deduceAdditionalLinks: bool = False
    ):
        """
        Args:
            feedbackSource (AbsFeedbackSource): source to which to ask for preferences
                for action data
            deduceAdditionalLinks (bool, optional): If true will create additional links
            between nodes in a single path. Defaults to False.

        """

        self.feedbackSource = feedbackSource

        self.deduceAdditionalLinks = deduceAdditionalLinks

    def add_preference_relationship_to_graphs(
        self,
        prefG: nx.DiGraph,
        genG: nx.Graph,
        node1: int,
        node2: int,
        preference: PreferencePairsData,
    ):
        """adds new connection to both preference graph and general (history) graph
        based on preference for given nodes pair

        Args:
            prefG (nx.DiGraph): graph storing preference relationships between nodes
            genG (nx.Graph): graph storing history of all node pairs for which
                a preference was asked for
            node1 (int): node corresponding to action1 in the action pair
            node2 (int): node corresponding to action2 in the action pair
            preference (PreferencePairsData): preference between action1 and action2

        Raises:
            Exception: if given more than 1 preference for a given pair
            Exception: if given preference has unexpected value
        """
        # from unprefered to prefered

        pref_tensor = preference.preference_pairs.cpu().detach()

        if pref_tensor.shape[0] != 1:
            raise Exception(
                f"Only one preference is expected for AddPreferenceToGraphs, found: {pref_tensor.shape[0]}"
            )

        preference = pref_tensor.tolist()[0]

        if preference == [0.5, 0.5]:
            # add links in two directions for deducing additional links
            prefG.add_edge(u_of_edge=node1, v_of_edge=node2, label="equal1")
            prefG.add_edge(u_of_edge=node2, v_of_edge=node1, label="equal2")
            genG.add_edge(u_of_edge=node1, v_of_edge=node2, label="base")
        elif preference == [1.0, 0.0]:
            prefG.add_edge(u_of_edge=node1, v_of_edge=node2, label="preferable")
            genG.add_edge(u_of_edge=node1, v_of_edge=node2, label="base")
        elif preference == [0.0, 1.0]:
            prefG.add_edge(u_of_edge=node2, v_of_edge=node1, label="preferable")
            genG.add_edge(u_of_edge=node1, v_of_edge=node2, label="base")
        elif preference == [0.0, 0.0]:
            genG.add_edge(u_of_edge=node1, v_of_edge=node2, label="zero")
        else:
            raise Exception(f"Unexpected preference data: {preference}")

    def find_edge_to_ask(
        self,
        actions_tensor: torch.tensor,
        edge_list: list,
        prefGraph: nx.DiGraph,
        genGraph: nx.Graph,
    ) -> bool:
        """Provided an edge list of possible connections, it will look for the first edge not present in the
        general graph and generate preference for actions corresponding to the edge nodes. Then
        it will add the new edge to the preferece and the general gragh in a way based on the received
        preference.

        Args:
            actions_tensor (torch.tensor): list of actions for which to generate preferences
            edge_list (list): list of node pairs, used as list of action pairs, which are
                candidates to be added to preference graph
            prefG (nx.DiGraph): graph storing preference relationships between nodes
            genG (nx.Graph): graph storing history of all node pairs for which
                a preference was asked for

        Returns:
            bool: true if the method had found an edge in the edge list avaliable to add to the
                preference graph. false otherwise
        """

        for node1, node2 in edge_list:

            if not genGraph.has_edge(u=node1, v=node2):

                action1 = actions_tensor[node1].unsqueeze(0)
                action2 = actions_tensor[node2].unsqueeze(0)

                action_pair_data = ActionPairsData(
                    action_pairs=torch.stack([action1, action2], dim=1)
                )

                preference_data = self.feedbackSource.generate_feedback(
                    action_pair_data
                )

                self.add_preference_relationship_to_graphs(
                    prefG=prefGraph,
                    genG=genGraph,
                    node1=node1,
                    node2=node2,
                    preference=preference_data,
                )

                return True

        return False

    @override
    def generate_preference_data(
        self, data: ActionData, limit: int
    ) -> tuple[ActionPairsData, PreferencePairsData]:
        """Constructs a directed graph of preferences relationships between all provided actions.
        First it tries to connect all the
        components together (edges with preferences [0,0] are ignored). If there is already one
        component or it is impossible to create it the strategy changes to connecting pairs of actions
        that have the smallest degrees in preference graph.

        A bidirectional graph is used in parralel to track already asked pairs and track
        pairs with [0,0] preferences.

        At the end the preference graph and the general bidirectional graph are used
        to generate action pairs and the corresponding preference data

        Args:
            data (ActionData): list of actions for which we want preference feedback
            limit (int): maximum number of preference method can ask the feedback source

        Raises:
            Exception: if an edge in the preference graph has an unknown edge label
            Exception: if an edge in the general graph has an unknown edge label

        Returns:
            tuple[ActionPairsData, PreferencePairsData]: list of action pairs with corresponding preferences
        """
        actions_tensor = data.actions

        general_graph = nx.Graph()
        preference_graph = nx.DiGraph()

        for i in range(actions_tensor.shape[0]):
            general_graph.add_node(i)
            preference_graph.add_node(i)

        max_graph_edges = len(nx.complete_graph(n=general_graph.nodes).edges)

        ask_counter = 0

        # until we get the target preference number or all the possible pairs
        # are asked
        while (ask_counter < limit) and (len(general_graph.edges) != max_graph_edges):

            gained_feedback = False

            components = list(nx.connected_components(nx.Graph(preference_graph)))

            # while there are several components in preference graph
            if len(components) > 1:
                components_lengths = np.array([len(c) for c in components])
                components_connect_order = np.argsort(components_lengths)
                components_combinations = combinations(
                    components_connect_order.tolist(), 2
                )

                # go through all present components and attempt to connect them
                for comp1_id, comp2_id in components_combinations:
                    if gained_feedback:
                        break

                    comp1 = components[comp1_id]
                    comp2 = components[comp2_id]

                    gained_feedback = self.find_edge_to_ask(
                        actions_tensor=actions_tensor,
                        edge_list=product(comp1, comp2),
                        prefGraph=preference_graph,
                        genGraph=general_graph,
                    )

                    if gained_feedback:
                        ask_counter += 1

                # attempt to connect two components

            # if we didn't ask for a preference feedback while connecting components
            if not gained_feedback:

                nodes = sorted(
                    dict(preference_graph.degree).keys(),
                    key=lambda x: preference_graph.degree[x],
                )

                nodes_combinations = combinations(nodes, 2)

                gained_feedback = self.find_edge_to_ask(
                    actions_tensor=actions_tensor,
                    edge_list=nodes_combinations,
                    prefGraph=preference_graph,
                    genGraph=general_graph,
                )

                if gained_feedback:
                    ask_counter += 1

        # if we so desire, generate additional links in the preference graph
        if self.deduceAdditionalLinks:
            # Use deep forward search to find all the nodes, that
            # are present in all possible paths, starting with
            # the analysed node, and create links between
            # the start node and all path nodes
            for node in preference_graph.nodes:
                DFS_res = list(nx.dfs_edges(preference_graph, source=node))
                DFS_nodes = set(sum([list(a) for a in DFS_res], start=[]))

                if node in DFS_nodes:
                    DFS_nodes.remove(node)

                for n in DFS_nodes:
                    preference_graph.add_edge(node, n)

        all_edges = general_graph.edges(data=True)
        all_directed_edges = preference_graph.edges(data=True)

        action_pair_tensor_list = []
        preference_pair_tensor_list = []

        # generate preference and actions from preference graph
        for a, b, m in all_directed_edges:
            action1 = actions_tensor[a].unsqueeze(0)
            action2 = actions_tensor[b].unsqueeze(0)

            if m["label"] == "equal1":
                action_pair_tensor_list.append(torch.stack([action1, action2], dim=1))
                preference_pair_tensor_list.append(torch.tensor([0.5, 0.5]))
            elif m["label"] == "equal2":
                pass
            elif m["label"] == "preferable":
                action_pair_tensor_list.append(torch.stack([action1, action2], dim=1))
                preference_pair_tensor_list.append(torch.tensor([0.0, 1.0]))
            else:
                raise Exception(f"Unknown label: {m['label']}")

        # generate preference and actions from general graph ([0,0] cases)
        for a, b, m in all_edges:
            if m["label"] == "zero":
                action1 = actions_tensor[a].unsqueeze(0)
                action2 = actions_tensor[b].unsqueeze(0)

                action_pair_tensor_list.append(torch.stack([action1, action2], dim=1))

                preference_pair_tensor_list.append(torch.tensor([0.0, 0.0]))
            elif m["label"] == "base":
                pass
            else:
                raise Exception(f"Unknown label: {m['label']}")

        action_pairs_data = ActionPairsData(
            action_pairs=torch.concat(action_pair_tensor_list, dim=0)
        )
        pref_pairs_data = PreferencePairsData(
            preference_pairs=torch.stack(preference_pair_tensor_list, dim=0)
        )

        return action_pairs_data, pref_pairs_data

    def __str__(self) -> str:
        """Returns string describing the object

        Returns:
            str
        """
        return "Graph Preference Data Generator"
