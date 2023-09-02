import argparse
import os
import networkx as nx
import osmium as o
from .const import candi_highway_types


class OSM2RNHandler(o.SimpleHandler):

    def __init__(self, rn):
        super(OSM2RNHandler, self).__init__()
        self.candi_highway_types = candi_highway_types
        self.rn = rn
        self.eid = 0

    def way(self, w):
        if 'highway' in w.tags and w.tags['highway'] in self.candi_highway_types:
            raw_eid = w.id
            full_coords = []
            for n in w.nodes:
                full_coords.append((n.lat, n.lon))
            if 'oneway' in w.tags:
                edge_attr = {'eid': self.eid, 'coords': full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                rn.add_edge(full_coords[0], full_coords[-1], **edge_attr)
                self.eid += 1
            else:
                edge_attr = {'eid': self.eid, 'coords': full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                rn.add_edge(full_coords[0], full_coords[-1], **edge_attr)
                self.eid += 1

                reversed_full_coords = full_coords.copy()
                reversed_full_coords.reverse()

                edge_attr = {'eid': self.eid, 'coords': reversed_full_coords, 'raw_eid': raw_eid, 'highway': w.tags['highway']}
                rn.add_edge(reversed_full_coords[0], reversed_full_coords[-1], **edge_attr)
                self.eid += 1


def store_osm(rn, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    """nodes: [lat, lng]"""
    print('# of nodes:{}'.format(rn.number_of_nodes()))
    print('# of edges:{}'.format(rn.number_of_edges()))
    node_map = {k: idx for idx, k in enumerate(rn.nodes())}

    with open(os.path.join(target_path, 'nodeOSM.txt'), 'w+') as node:
        for idx, coord in enumerate(rn.nodes()):
            node.write(f'{"%-6d"%idx} {coord[0]} {coord[1]}\n')
        node.close()

    edges = {}
    for stcoord, encoord, data in rn.edges(data=True):
        edges[data['eid']] = {'st': node_map[stcoord],
                              'en': node_map[encoord],
                              'coords': data["coords"],
                              'type': data['highway']}

    with open(os.path.join(target_path, 'edgeOSM.txt'), 'w+') as edge:
        for idx, k in enumerate(sorted(edges.keys())):
            edge.write(f'{idx} {edges[k]["st"]} {edges[k]["en"]} {len(edges[k]["coords"])}')
            for coord in edges[k]["coords"]:
                edge.write(f' {coord[0]} {coord[1]}')
            edge.write('\n')
        edge.close()

    with open(os.path.join(target_path, 'wayTypeOSM.txt'), 'w+') as waytype:
        for idx, k in enumerate(sorted(edges.keys())):
            waytype.write(f'{"%-6d"%idx} {"%-10s"%edges[k]["type"]} {"%-4d"%candi_highway_types[edges[k]["type"]]}\n')
        waytype.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='the input path of the original osm data')
    parser.add_argument('--output_path', help='the output directory of the constructed road network')
    opt = parser.parse_args()
    print(opt)

    rn = nx.DiGraph()
    handler = OSM2RNHandler(rn)
    handler.apply_file(opt.input_path, locations=True)
    store_osm(rn, opt.output_path)
