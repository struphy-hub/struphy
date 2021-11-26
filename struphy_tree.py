from treelib import Node, Tree
import os
import struphy.diagnostics.diagn_tools as tools

nodes = dict()
parents = ['struphy']
subdirs = []
for root, dirs, files in os.walk(os.path.dirname(os.path.realpath(__file__))):

    if root.endswith('struphy'):
        nodes['struphy'] = {'name': 'struphy' + '/', 'parent': None}
        subdirs = dirs
        subdirs.append('__pyccel__')
        len_of_set = len(set(subdirs))
        continue
    elif '__pycache__' in root or '__pyccel__' in root or '.pytest_cache' in root or 'gvec_to_python' in root:
        continue

    for dir in subdirs:
        if dir in ['__pycache__', '__pyccel__']:
            continue
        elif root.endswith(dir):
            #print(root)
            if parents[-1] not in root:
                parents.pop()
            nodes[dir] = {'name': dir, 'parent': parents[-1]}
            
            subdirs.extend(dirs)
            if len(set(subdirs)) > len_of_set:
                len_of_set = len(set(subdirs))
                parents.append(dir)
            #print('parents:', parents)

tree = Tree()
for key, val in nodes.items():
    #print(key, val)
    tree.create_node(val['name'], key, parent=val['parent'])
tree.show()




