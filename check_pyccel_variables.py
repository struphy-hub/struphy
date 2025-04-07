from struphy.utils.utils import read_state
import ast
import re

START_OMP_REGION = "#$ omp parallel"
END_OMP_REGION = "#$ omp end parallel"

def get_omp_regions(lines):
    regions = []
    for iline, line in enumerate(lines):
        if line[:3] == 'def':
            pattern_fname = r'^def\s+(\w+)\s*\('
            match = re.search(pattern_fname, line)
            function_name = match.group(1)
        if START_OMP_REGION in line:
            start = iline
        if END_OMP_REGION in line:
            end = iline
            regions.append({
                'function': function_name,
                'omp_region': lines[start:end+1],
                })
    return regions


def get_omp_variables(lines):
    # Pattern to capture variable names inside each clause
    pattern = r'(private|firstprivate|shared)\s*\(([^)]+)\)'
    variables = {
        'private': [],
        'firstprivate': [],
        'shared': []
    }
    
    for line in lines:
        if "#$ omp" in line:
            matches = re.findall(pattern, line)
            for kind, varlist in matches:
                variables[kind].extend([v.strip() for v in varlist.split(',')])
    return variables

def remove_tabbing(lines):
    line0 = lines[0]
    num_spaces = len(line0) - len(line0.lstrip(' '))
    # print("Number of leading spaces:", num_spaces)
    lines_out = []
    for line in lines:
        lines_out.append(line[num_spaces:])
    return lines_out

def get_python_variables(lines):
    lines = remove_tabbing(lines)

    # Parse the source into an AST
    source = ''.join(lines)
    tree = ast.parse(source)

    # Define a visitor to collect all variable names
    class VariableCollector(ast.NodeVisitor):
        def __init__(self):
            self.names = set()
        
        def visit_Name(self, node):
            # Add the name (node.id) to the set
            self.names.add(node.id)
            self.generic_visit(node)

    # Create the visitor and walk the tree
    collector = VariableCollector()
    collector.visit(tree)

    #print("All variable names:", collector.names)
    return collector.names

def check_omp_region(omp_region, exit_on_fail = False):
    omp_variables = get_omp_variables(omp_region)
    python_variables = get_python_variables(omp_region)
    for variable in python_variables:
        declared = False
        for key, omp_var_list in omp_variables.items():
            if variable in omp_var_list:
                declared = True
                print(f"{variable} is {key}!")
        if not declared:
            print(f"{variable} is undeclared" + "!" * 50)
            if exit_on_fail:
                exit()

def check_file(filename, exit_on_fail = False):
    print(f"# Checking file: {filename}")
    with open(filename, 'r') as file:
        lines = file.readlines()
    omp_region_dicts = get_omp_regions(lines)

    for omp_region_dict in omp_region_dicts:
        function = omp_region_dict['function']
        print(f"## {function}")
        omp_region = omp_region_dict['omp_region']
        check_omp_region(omp_region, exit_on_fail)

state = read_state()
for kernel in state['kernels']:
    check_file(kernel, exit_on_fail = False)
