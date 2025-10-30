import importlib.metadata
import re
import tomllib

import tomli_w


def get_min_bound(entry):
    match = re.search(r"(>=|==|~=|>|>)\s*([\w\.\-]+)", entry)
    if match:
        op, version = match.groups()
        return f"{op}{version}"
    return None


def get_max_bound(entry):
    match = re.search(r"(<=|<)\s*([\w\.\-]+)", entry)
    if match:
        op, version = match.groups()
        return f"{op}{version}"
    return None


def get_package_name(entry):
    return re.split(r"[<>=~]", entry.strip())[0].replace(" ", "")


def generate_updated_entry(package_name, package_deps):
    ver_def = package_name
    # Always set max version to the currently installed version
    ver_def += f"<={package_deps['installed']}"

    if package_deps["min"]:
        ver_def += f", {package_deps['min']}"
    return ver_def


def update_dependencies(dependencies):
    for i, entry in enumerate(dependencies):
        package_name = get_package_name(entry)

        try:
            installed_version = importlib.metadata.version(package_name)

            package_deps = {"installed": installed_version, "min": get_min_bound(entry), "max": get_max_bound(entry)}

            if package_deps["installed"]:
                dependencies[i] = generate_updated_entry(package_name, package_deps)

        except importlib.metadata.PackageNotFoundError:
            print(f"Warning: {package_name} not installed, skipping...")
            continue

    # Remove psydac from the dependencies
    for i, entry in enumerate(dependencies):
        if "psydac" in entry:
            dependencies.pop(i)


def main():
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)

    mandatory_dependencies = pyproject_data["project"]["dependencies"]
    optional_dependency_groups = pyproject_data["project"]["optional-dependencies"]

    update_dependencies(mandatory_dependencies)
    for group_name, group_deps in optional_dependency_groups.items():
        update_dependencies(group_deps)

    with open("pyproject.toml", "wb") as f:
        tomli_w.dump(pyproject_data, f)


if __name__ == "__main__":
    main()
