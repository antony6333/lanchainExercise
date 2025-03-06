default_file_path = "config.properties"
def read_properties(file_path=None):
    if file_path is None:
        file_path = default_file_path
    props = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # 忽略空行和註釋
                key, value = line.split('=', 1)
                props[key.strip()] = value.strip()
    return props

def get_property_value(key):
    return read_properties().get(key)